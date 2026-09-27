"""
Motor matemático de Blackjack basado en composición exacta del zapato.

Ideas clave (inspiradas en el analizador combinatorio de possibly-wrong/blackjack)
-------------------------------------------------------------------------------
* La baraja es un vector de 10 contadores: A, 2..9, T (T/J/Q/K valen 10).
* Cada nodo del árbol del jugador es el multiconjunto de cartas que ha robado; para
  cada nodo hace falta la distribución del crupier *con esas cartas retiradas*
  (efecto de eliminación exacto, como las probabilidades "sobre el subconjunto
  del zapato" del analizador de referencia).
* La distribución del crupier se calcula con programación dinámica memoizada sobre
  el multiconjunto que roba el crupier, vectorizada con NumPy: cada estado del
  crupier lleva un vector de probabilidades con una entrada por composición
  (nodo del jugador). Un único recorrido resuelve miles de composiciones.
* El EV del jugador se obtiene con recursión memoizada sobre su árbol (CDZ-:
  la estrategia es óptima para la composición de cada nodo).
* Hole card: la carta oculta del crupier se condiciona a "no blackjack" y el EV
  resultante es el EV *dado que el crupier no tiene blackjack*. Con peek (US) ése
  es el EV en el momento de decidir. Con ENHC (European No Hole Card) el crupier
  aún puede tener blackjack y se lleva todo lo apostado (también lo doblado y
  separado): EV_enhc = (1 - p_bj) * EV_cond - p_bj * apuesta_total.

Aproximaciones documentadas
---------------------------
* Split: dos manos independientes que parten del mismo zapato, sin re-split.
* Las cartas que roba el jugador usan el zapato sin condicionar la carta oculta.

Modos de cálculo
----------------
* "exact": recursión matemática pura.
* "ml":    inferencia con `blackjack_model.pkl` (entrenado con train_model.py).
* "auto":  ML si el modelo existe, si no, exacto.
"""
from __future__ import annotations

import math
import os
import threading
import time
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from features import OUTCOME_FEATURES, RANK_LABELS, build_features, to_vector

OUTCOMES: tuple[str, ...] = ("17", "18", "19", "20", "21", "bust")
ACE, TEN = 0, 9
MODEL_PATH = Path(os.environ.get("BJ_MODEL_PATH", Path(__file__).parent / "blackjack_model.pkl"))

_ALIASES = {"A": ACE, "1": ACE, "11": ACE, "T": TEN, "10": TEN, "J": TEN, "Q": TEN, "K": TEN}
_ALIASES.update({str(v): v - 1 for v in range(2, 10)})

ACTION_ES = {
    "Stand": "Plantarse",
    "Hit": "Pedir",
    "Double": "Doblar",
    "Split": "Separar",
    "Surrender": "Rendirse",
    "Blackjack": "Blackjack",
    "Bust": "Pasado",
}
# Apuesta total expuesta a un blackjack del crupier en ENHC.
_STAKE = {"Stand": 1.0, "Hit": 1.0, "Double": 2.0, "Split": 2.0, "Surrender": 1.0}


def rank_index(card: str | int) -> int:
    """Convierte 'A', '2'..'9', 'T', 'J', 'Q', 'K', '10' al índice 0..9."""
    key = str(card).strip().upper()
    if key not in _ALIASES:
        raise ValueError(f"Carta no válida: {card!r}")
    return _ALIASES[key]


def hand_total(indices: Iterable[int]) -> tuple[int, bool]:
    """Devuelve (total, es_blanda) de una mano dada como índices de rango."""
    hard, has_ace = 0, False
    for r in indices:
        hard += r + 1
        has_ace = has_ace or r == ACE
    if has_ace and hard + 10 <= 21:
        return hard + 10, True
    return hard, False


# --------------------------------------------------------------------------- #
# Estado del zapato
# --------------------------------------------------------------------------- #
class DeckState:
    """Cartas que quedan en el zapato, por valor (A, 2..9, T)."""

    def __init__(self, num_decks: int = 6, counts: Sequence[int] | None = None):
        if num_decks < 1:
            raise ValueError("El número de barajas debe ser >= 1")
        self.num_decks = num_decks
        self.initial = [4 * num_decks] * 9 + [16 * num_decks]
        self.counts = list(counts) if counts is not None else list(self.initial)

    def copy(self) -> "DeckState":
        return DeckState(self.num_decks, self.counts)

    @property
    def total(self) -> int:
        return sum(self.counts)

    @property
    def initial_total(self) -> int:
        return 52 * self.num_decks

    def remove(self, card: str | int) -> None:
        idx = card if isinstance(card, int) else rank_index(card)
        if self.counts[idx] <= 0:
            raise ValueError(f"No quedan cartas '{RANK_LABELS[idx]}' en el zapato")
        self.counts[idx] -= 1

    def remove_many(self, cards: Iterable[str | int]) -> None:
        for c in cards:
            self.remove(c)

    def add(self, card: str | int) -> None:
        idx = card if isinstance(card, int) else rank_index(card)
        if self.counts[idx] >= self.initial[idx]:
            raise ValueError(f"El zapato ya tiene todas las cartas '{RANK_LABELS[idx]}'")
        self.counts[idx] += 1

    def as_tuple(self) -> tuple[int, ...]:
        return tuple(self.counts)

    def summary(self) -> dict:
        """Composición, penetración y conteo Hi-Lo de las cartas vistas."""
        seen = [i - c for i, c in zip(self.initial, self.counts)]
        running = sum(seen[1:6]) - seen[ACE] - seen[TEN]
        decks_left = self.total / 52
        return {
            "num_decks": self.num_decks,
            "remaining": {lbl: c for lbl, c in zip(RANK_LABELS, self.counts)},
            "initial": {lbl: c for lbl, c in zip(RANK_LABELS, self.initial)},
            "total": self.total,
            "initial_total": self.initial_total,
            "decks_remaining": round(decks_left, 2),
            "penetration": round(1 - self.total / self.initial_total, 4),
            "running_count": running,
            "true_count": round(running / decks_left, 2) if decks_left > 0 else 0.0,
        }


# --------------------------------------------------------------------------- #
# Reglas
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Rules:
    num_decks: int = 6
    dealer_hits_soft17: bool = False  # False = S17 (el crupier se planta en todos los 17)
    double_after_split: bool = True
    blackjack_payout: float = 1.5
    surrender: bool = False  # rendición tardía
    enhc: bool = False  # European No Hole Card (False = peek americano)

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Árbol del jugador
# --------------------------------------------------------------------------- #
@dataclass
class _PlayerTree:
    """Nodo = multiconjunto de cartas robadas desde la mano inicial."""

    keys: list[tuple[int, ...]]
    levels: list[int]
    totals: list[int]
    children: list[list[tuple[int, int]]]  # (cartas disponibles de ese rango, hijo o -1 si se pasa)
    n0: int  # cartas en el zapato en la raíz
    dealer: np.ndarray = None  # (6, K) distribución del crupier por nodo
    stand: list[float] = None
    win: list[float] = None  # P(ganar) plantándose en cada nodo
    push: list[float] = None


class _Solver:
    """EV y distribución Win/Push/Loss memoizados sobre un árbol del jugador."""

    def __init__(self, tree: _PlayerTree):
        self.t = tree
        self._best: dict[int, float] = {}
        self._hit: dict[int, float] = {}
        self._dist: dict[int, tuple[float, float, float]] = {}

    def hit(self, i: int) -> float:
        v = self._hit.get(i)
        if v is None:
            t = self.t
            kids, n = t.children[i], t.n0 - t.levels[i]
            if not kids or n <= 0:
                v = -math.inf
            else:
                v = sum(a * (self.best(j) if j >= 0 else -1.0) for a, j in kids) / n
            self._hit[i] = v
        return v

    def best(self, i: int) -> float:
        v = self._best.get(i)
        if v is None:
            v = self._best[i] = max(self.t.stand[i], self.hit(i))
        return v

    def double(self, i: int) -> float:
        t = self.t
        kids, n = t.children[i], t.n0 - t.levels[i]
        if not kids or n <= 0:
            return -math.inf
        return 2.0 * sum(a * (t.stand[j] if j >= 0 else -1.0) for a, j in kids) / n

    # ---- distribuciones (win, push, loss) siguiendo la política óptima ---- #
    def stand_dist(self, i: int) -> tuple[float, float, float]:
        w, p = self.t.win[i], self.t.push[i]
        return w, p, 1.0 - w - p

    def _mix(self, i: int, leaf) -> tuple[float, float, float]:
        t = self.t
        n = t.n0 - t.levels[i]
        w = p = l = 0.0
        for a, j in t.children[i]:
            if j < 0:
                l += a
            else:
                dw, dp, dl = leaf(j)
                w, p, l = w + a * dw, p + a * dp, l + a * dl
        return w / n, p / n, l / n

    def hit_dist(self, i: int) -> tuple[float, float, float]:
        return self._mix(i, self.best_dist)

    def best_dist(self, i: int) -> tuple[float, float, float]:
        d = self._dist.get(i)
        if d is None:
            d = self._dist[i] = self.stand_dist(i) if self.t.stand[i] >= self.hit(i) else self.hit_dist(i)
        return d

    def double_dist(self, i: int) -> tuple[float, float, float]:
        return self._mix(i, self.stand_dist)


# --------------------------------------------------------------------------- #
# Modelo de ML (opcional)
# --------------------------------------------------------------------------- #
class CompiledForest:
    """
    Gradient boosting "compilado" a arrays planos de NumPy.

    `HistGradientBoostingRegressor.predict` recorre ~600 árboles uno a uno y tarda
    varios ms por fila. Aquí se recorren todos los árboles a la vez, nivel a nivel,
    con operaciones vectorizadas: < 0,3 ms por acción y sin depender de sklearn.
    """

    def __init__(self, arrays: dict):
        self.feature = arrays["feature"]
        self.threshold = arrays["threshold"]
        self.left = arrays["left"]
        self.right = arrays["right"]
        self.is_leaf = arrays["is_leaf"]
        self.value = arrays["value"]
        self.roots = arrays["roots"]
        self.baseline = float(arrays["baseline"])
        self.max_depth = int(arrays["max_depth"])

    @staticmethod
    def arrays_from_hgb(model) -> dict:
        feature, threshold, left, right, is_leaf, value, roots = [], [], [], [], [], [], []
        offset, max_depth = 0, 0
        for predictors in model._predictors:
            nodes = predictors[0].nodes
            roots.append(offset)
            feature.append(nodes["feature_idx"].astype(np.int64))
            threshold.append(nodes["num_threshold"].astype(np.float64))
            left.append(nodes["left"].astype(np.int64) + offset)
            right.append(nodes["right"].astype(np.int64) + offset)
            is_leaf.append(nodes["is_leaf"].astype(bool))
            value.append(nodes["value"].astype(np.float64))
            max_depth = max(max_depth, int(nodes["depth"].max()))
            offset += len(nodes)
        return {
            "feature": np.concatenate(feature), "threshold": np.concatenate(threshold),
            "left": np.concatenate(left), "right": np.concatenate(right),
            "is_leaf": np.concatenate(is_leaf), "value": np.concatenate(value),
            "roots": np.array(roots, dtype=np.int64),
            "baseline": float(np.ravel(model._baseline_prediction)[0]), "max_depth": max_depth,
        }

    def predict_one(self, x: np.ndarray) -> float:
        idx = self.roots
        for _ in range(self.max_depth):
            leaf = self.is_leaf[idx]
            if leaf.all():
                break
            go_left = x[self.feature[idx]] <= self.threshold[idx]
            idx = np.where(leaf, idx, np.where(go_left, self.left[idx], self.right[idx]))
        return self.baseline + float(self.value[idx].sum())


class MLPredictor:
    """Carga `blackjack_model.pkl` bajo demanda y lo recarga si el archivo cambia."""

    def __init__(self, path: Path = MODEL_PATH):
        self.path = Path(path)
        self._bundle = None
        self._mtime: float | None = None
        self._error: str | None = None
        self._lock = threading.Lock()

    def bundle(self):
        try:
            mtime = self.path.stat().st_mtime
        except FileNotFoundError:
            self._bundle, self._mtime = None, None
            return None
        if mtime != self._mtime:
            with self._lock:
                if mtime != self._mtime:
                    try:
                        import joblib

                        bundle = joblib.load(self.path)
                        bundle["_compiled"] = {a: CompiledForest(arr) for a, arr in bundle.get("compiled", {}).items()}
                        self._bundle = bundle
                        self._error = None
                    except Exception as exc:  # modelo corrupto o sklearn ausente -> fallback exacto
                        self._bundle, self._error = None, f"{type(exc).__name__}: {exc}"
                    self._mtime = mtime
        return self._bundle

    @property
    def available(self) -> bool:
        return self.bundle() is not None

    def status(self) -> dict:
        b = self.bundle()
        if b is None:
            return {"available": False, "path": str(self.path), "error": self._error}
        return {
            "available": True,
            "path": str(self.path),
            "trained_at": b.get("trained_at"),
            "metrics": b.get("metrics"),
            "n_samples": b.get("n_samples"),
        }

    def predict_evs(self, feats: dict, actions: Sequence[str]) -> dict[str, float]:
        b = self.bundle()
        x = np.array(to_vector(feats, b["feature_names"]))
        out = {}
        for a in actions:
            if a in b["_compiled"]:
                out[a] = b["_compiled"][a].predict_one(x)
            elif a in b["ev_models"]:
                out[a] = float(b["ev_models"][a].predict(x[None, :])[0])
        return out

    def predict_outcome(self, feats: dict) -> dict[str, float] | None:
        b = self.bundle()
        if b is None:
            return None
        table = b.get("outcome_table")
        if table is not None:
            key = tuple(int(feats[f]) for f in OUTCOME_FEATURES)
            return table.get(key)
        clf = b.get("outcome_model")
        if clf is None:
            return None
        x = np.array([to_vector(feats, b.get("outcome_features", OUTCOME_FEATURES))])
        return {str(c): float(p) for c, p in zip(clf.classes_, clf.predict_proba(x)[0])}


PREDICTOR = MLPredictor()


# --------------------------------------------------------------------------- #
# Ventaja y Kelly
# --------------------------------------------------------------------------- #
# Efectos de eliminación (Griffin, "The Theory of Blackjack"): cambio de la ventaja
# del jugador, en %, al retirar una carta de ese valor de una baraja de 52.
EFFECT_OF_REMOVAL = {"A": -0.61, "2": 0.38, "3": 0.44, "4": 0.55, "5": 0.69,
                     "6": 0.46, "7": 0.28, "8": 0.00, "9": -0.18, "T": -0.51}
_DECK_ADJ = {1: 0.0048, 2: 0.0019, 3: 0.0010, 4: 0.0006, 5: 0.0002, 6: 0.0, 7: -0.0001, 8: -0.0002}
BLACKJACK_VARIANCE = 1.33  # varianza por mano (desv. típica ~1.15 unidades)


def base_player_edge(rules: Rules) -> float:
    """Ventaja del jugador con zapato neutro y estrategia básica (aprox. publicada)."""
    edge = -0.0041  # 6 barajas, S17, DAS, peek, BJ 3:2, sin rendición
    edge += _DECK_ADJ.get(rules.num_decks, -0.0002)
    if rules.dealer_hits_soft17:
        edge -= 0.0022
    if not rules.double_after_split:
        edge -= 0.0014
    if rules.enhc:
        edge -= 0.0011
    if rules.surrender:
        edge += 0.0008
    edge += (rules.blackjack_payout - 1.5) * 0.0453  # frecuencia de blackjack ~4.53 %
    return edge


def estimate_advantage(counts: Sequence[int], rules: Rules) -> dict:
    """
    Ventaja del jugador para la próxima mano a partir de la composición exacta:
    ventaja base de las reglas + suma lineal de efectos de eliminación, escalando
    la desviación de cada valor respecto a un zapato neutro a "cartas de una baraja".
    """
    n = sum(counts)
    base = base_player_edge(rules)
    if n == 0:
        return {"advantage": base, "base_edge": base, "shift": 0.0}
    shift = 0.0
    for lbl, c in zip(RANK_LABELS, counts):
        neutral = (16 if lbl == "T" else 4) / 52
        shift += EFFECT_OF_REMOVAL[lbl] / 100 * 52 * (neutral - c / n)
    return {"advantage": base + shift, "base_edge": base, "shift": shift}


def kelly_bet(advantage: float, bankroll: float, fraction: float = 0.5,
              min_bet: float = 10.0, max_bet: float | None = None) -> dict:
    """Kelly fraccional: f* = ventaja / varianza; apuesta = bankroll * fracción * f*."""
    full = max(advantage, 0.0) / BLACKJACK_VARIANCE
    raw = bankroll * fraction * full
    bet = max(raw, min_bet)
    if max_bet:
        bet = min(bet, max_bet)
    return {
        "advantage": advantage,
        "full_kelly_fraction": full,
        "kelly_fraction": fraction,
        "optimal_bet": raw,
        "recommended_bet": round(bet, 2),
        "at_minimum": raw <= min_bet,
        "bankroll": bankroll,
    }


# --------------------------------------------------------------------------- #
# Calculadora
# --------------------------------------------------------------------------- #
class BlackjackCalculator:
    def __init__(self, rules: Rules | None = None):
        self.rules = rules or Rules()

    # ---------------------------- Crupier ---------------------------------- #
    def dealer_distribution(
        self, counts: Sequence[int], up: int, removals: np.ndarray | Sequence[Sequence[int]]
    ) -> np.ndarray:
        """
        Probabilidades finales del crupier (17, 18, 19, 20, 21, bust), condicionadas
        a que no tenga blackjack, para cada composición `counts - removals[k]`.
        Devuelve un array (6, K).

        DP memoizada: el estado es el multiconjunto que ha robado el crupier (fija su
        total y si es blando). Estados alcanzados por órdenes distintos se fusionan
        (la probabilidad sólo depende del multiconjunto) y cada estado se expande una
        sola vez en lote para las K composiciones.
        """
        R = np.asarray(removals, dtype=np.float64).reshape(-1, 10)
        K = R.shape[0]
        c = np.asarray(counts, dtype=np.float64)
        MT = c[:, None] - R.T  # (10, K) cartas disponibles por rango y composición
        N = MT.sum(axis=0)
        out = np.zeros((6, K))

        h17 = self.rules.dealer_hits_soft17
        excluded = TEN if up == ACE else ACE if up == TEN else None  # carta oculta prohibida

        # Transiciones por (total duro, tiene_as): matriz 6x10 que lleva cada carta que
        # termina la mano a su resultado, y lista de cartas que dejan al crupier pidiendo.
        transitions: dict[tuple[int, bool], tuple[np.ndarray, list[tuple[int, int, bool]]]] = {}

        def transition(hard: int, has_ace: bool):
            t = transitions.get((hard, has_ace))
            if t is None:
                term = np.zeros((6, 10))
                cont = []
                for r in range(10):
                    nh = hard + r + 1
                    ace = has_ace or r == ACE
                    soft = ace and nh + 10 <= 21
                    total = nh + 10 if soft else nh
                    if nh > 21:
                        term[5, r] = 1.0
                    elif total >= 17 and not (h17 and soft and total == 17):
                        term[total - 17, r] = 1.0
                    else:
                        cont.append((r, nh, ace))
                t = transitions[(hard, has_ace)] = (term, cont)
            return t

        # estado: multiconjunto robado -> (total duro, tiene_as, vector de probabilidad)
        level: dict[tuple[int, ...], tuple[int, bool, np.ndarray]] = {
            (0,) * 10: (up + 1, up == ACE, np.ones(K))
        }
        drawn = 0
        while level:
            nxt: dict[tuple[int, ...], tuple[int, bool, np.ndarray]] = {}
            first = drawn == 0 and excluded is not None
            denom = N - MT[excluded] if first else N - drawn
            inv = 1.0 / np.where(denom > 0, denom, 1.0)
            for d, (hard, has_ace, pv) in level.items():
                # Filas imposibles (d > disponibles) ya tienen pv = 0, no hace falta recortar.
                probs = (MT - np.asarray(d, dtype=np.float64)[:, None]) * (pv * inv)
                if first:
                    probs[excluded] = 0.0
                term, cont = transition(hard, has_ace)
                out += term @ probs
                for r, nh, ace in cont:
                    if c[r] - d[r] <= 0 or (first and r == excluded):
                        continue
                    key = d[:r] + (d[r] + 1,) + d[r + 1 :]
                    entry = nxt.get(key)
                    if entry is None:
                        nxt[key] = (nh, ace, probs[r])
                    else:
                        entry[2].__iadd__(probs[r])
            level = nxt
            drawn += 1
        return out

    @staticmethod
    def dealer_blackjack_prob(counts: Sequence[int], up: int) -> float:
        n = sum(counts)
        if n == 0:
            return 0.0
        if up == ACE:
            return counts[TEN] / n
        if up == TEN:
            return counts[ACE] / n
        return 0.0

    # ---------------------------- Jugador ---------------------------------- #
    @staticmethod
    def _stand_outcomes(totals: np.ndarray, dealer: np.ndarray):
        """P(ganar) y P(empatar) plantándose, para cada nodo."""
        K = dealer.shape[1]
        cz = np.vstack([np.zeros(K), np.cumsum(dealer[:5], axis=0)])  # cz[j] = P(crupier < 17+j)
        cols = np.arange(K)
        less = cz[np.clip(totals - 17, 0, 5), cols]
        greater = cz[5] - cz[np.clip(totals - 16, 0, 5), cols]
        win = dealer[5] + less
        push = 1.0 - win - greater
        return win, push, win - greater

    @staticmethod
    def _build_tree(counts: Sequence[int], hand: Sequence[int], max_draws: int | None = None) -> _PlayerTree:
        hard0 = sum(r + 1 for r in hand)
        root = (0,) * 10
        index = {root: 0}
        keys, hards, aces, levels = [root], [hard0], [ACE in hand], [0]
        children: list[list[tuple[int, int]]] = []

        i = 0
        while i < len(keys):
            R, h = keys[i], hards[i]
            kids: list[tuple[int, int]] = []
            if h < 21 and (max_draws is None or levels[i] < max_draws):
                for r in range(10):
                    avail = counts[r] - R[r]
                    if avail <= 0:
                        continue
                    nh = h + r + 1
                    if nh > 21:
                        kids.append((avail, -1))
                        continue
                    child = R[:r] + (R[r] + 1,) + R[r + 1 :]
                    j = index.get(child)
                    if j is None:
                        j = len(keys)
                        index[child] = j
                        keys.append(child)
                        hards.append(nh)
                        aces.append(aces[i] or r == ACE)
                        levels.append(levels[i] + 1)
                    kids.append((avail, j))
            children.append(kids)
            i += 1

        totals = [h + 10 if a and h + 10 <= 21 else h for h, a in zip(hards, aces)]
        return _PlayerTree(keys, levels, totals, children, sum(counts))

    def _attach_dealer(self, counts: Sequence[int], up: int, *trees: _PlayerTree) -> None:
        """Una sola pasada del crupier para la unión de composiciones de todos los árboles."""
        union: dict[tuple[int, ...], int] = {}
        for tree in trees:
            for k in tree.keys:
                union.setdefault(k, len(union))
        dist = self.dealer_distribution(counts, up, np.array(list(union)))
        for tree in trees:
            tree.dealer = dist[:, [union[k] for k in tree.keys]]
            win, push, ev = self._stand_outcomes(np.array(tree.totals), tree.dealer)
            tree.stand, tree.win, tree.push = ev.tolist(), win.tolist(), push.tolist()

    def _split(self, tree: _PlayerTree, pair: int) -> tuple[float, tuple[float, float, float]]:
        """EV de separar (2 manos independientes) y distribución W/P/L por mano."""
        s = _Solver(tree)
        das = self.rules.double_after_split
        ev = w = p = l = 0.0
        for a, j in tree.children[0]:
            if pair == ACE:  # ases separados: una carta y plantarse
                v, d = tree.stand[j], s.stand_dist(j)
            else:
                v, d = s.best(j), s.best_dist(j)
                if das and s.double(j) > v:
                    v, d = s.double(j), s.double_dist(j)
            ev += a * v
            w, p, l = w + a * d[0], p + a * d[1], l + a * d[2]
        n = tree.n0
        return 2.0 * ev / n, (w / n, p / n, l / n)

    # ---------------------------- Núcleo exacto ---------------------------- #
    def solve_exact(self, counts: tuple[int, ...], player: tuple[int, ...], up: int) -> dict:
        """
        EVs condicionados a "el crupier no tiene blackjack" y distribuciones W/P/L.
        No aplica ajustes de ENHC ni blackjack natural (lo hace `analyze`).
        """
        # el orden de las cartas no altera el EV: normalizar mejora la tasa de acierto de la caché
        return _solve_exact_cached(self.rules, tuple(counts), tuple(sorted(player)), up)

    def _solve_exact(self, counts, player, up) -> dict:
        tree = self._build_tree(counts, player)
        split_tree = None
        if len(player) == 2 and player[0] == player[1]:
            split_tree = self._build_tree(counts, player[:1], max_draws=1 if player[0] == ACE else None)
            self._attach_dealer(counts, up, tree, split_tree)
        else:
            self._attach_dealer(counts, up, tree)
        s = _Solver(tree)

        evs = {"Stand": tree.stand[0], "Hit": s.hit(0)}
        dists = {"Stand": s.stand_dist(0)}
        if evs["Hit"] != -math.inf:
            dists["Hit"] = s.hit_dist(0)
        else:
            del evs["Hit"]
        if len(player) == 2:
            evs["Double"], dists["Double"] = s.double(0), s.double_dist(0)
            if split_tree is not None:
                evs["Split"], dists["Split"] = self._split(split_tree, player[0])
        return {"evs": evs, "dists": dists, "dealer": tree.dealer[:, 0].tolist()}

    # ---------------------------- Análisis completo ------------------------ #
    def analyze(self, counts: Sequence[int], player: Sequence[int], up: int | None,
                mode: str = "auto", compare_basic: bool = True) -> dict:
        """
        counts: zapato restante (sin cartas muertas, mano del jugador ni carta visible).
        player: índices de las cartas del jugador. up: índice de la carta del crupier.
        mode: "auto" | "exact" | "ml".
        """
        t0 = time.perf_counter()
        counts = tuple(int(x) for x in counts)
        player = tuple(player)
        total, soft = hand_total(player)
        rules = self.rules
        res: dict = {
            "player": {
                "cards": [RANK_LABELS[r] for r in player],
                "total": total,
                "soft": soft,
                "pair": len(player) == 2 and player[0] == player[1],
                "blackjack": len(player) == 2 and total == 21,
                "busted": total > 21,
            },
            "dealer": None,
            "actions": [],
            "best": None,
            "explanation": None,
            "basic_strategy": None,
            "insurance": None,
            "message": None,
            "source": None,
        }
        n = sum(counts)
        if n == 0:
            res["message"] = "El zapato está vacío."
            return res
        if up is None:
            res["message"] = "Introduce la carta visible del crupier."
            return res

        p_bj = self.dealer_blackjack_prob(counts, up)
        res["dealer"] = {"upcard": RANK_LABELS[up], "blackjack_prob": p_bj}
        if up == ACE:
            p_ten = counts[TEN] / n
            res["insurance"] = {"ten_prob": p_ten, "ev": 3 * p_ten - 1, "recommended": p_ten > 1 / 3}

        if len(player) < 2 or total > 21:
            res["dealer"]["probs"] = self._dealer_probs(counts, up)
            if total > 21:
                res["actions"] = [{"action": "Bust", "ev": -1.0}]
                res["best"] = "Bust"
                res["message"] = "Te has pasado (bust)."
            else:
                res["message"] = "Introduce al menos dos cartas propias."
            return res

        use_ml = mode == "ml" or (mode == "auto" and PREDICTOR.available)
        if use_ml and not PREDICTOR.available:
            res["message"] = "No hay modelo entrenado: usando cálculo exacto."
            use_ml = False

        if res["player"]["blackjack"]:
            res["dealer"]["probs"] = self._dealer_probs(counts, up)
            ev = rules.blackjack_payout * (1 - p_bj)
            res["actions"] = [{"action": "Blackjack", "ev": ev, "win": 1 - p_bj, "push": p_bj, "loss": 0.0}]
            res["best"] = "Blackjack"
            res["source"] = "exact"
            res["explanation"] = (
                f"Blackjack natural: cobras {rules.blackjack_payout:g}:1"
                + (f" salvo que el crupier también tenga blackjack ({p_bj:.1%})." if p_bj else ".")
            )
            return self._finish(res, t0)

        if use_ml:
            core = self._solve_ml(counts, player, up)
            res["source"] = "ml"
        else:
            core = self.solve_exact(counts, player, up)
            res["source"] = "exact"

        res["dealer"]["probs"] = dict(zip(OUTCOMES, core["dealer"]))
        evs = dict(core["evs"])
        dists = dict(core["dists"])
        if rules.surrender and len(player) == 2:
            evs["Surrender"] = -0.5
            dists["Surrender"] = (0.0, 0.0, 1.0)

        # ENHC: el blackjack del crupier se lleva toda la apuesta expuesta.
        if rules.enhc and p_bj > 0:
            for a in evs:
                evs[a] = (1 - p_bj) * evs[a] - p_bj * _STAKE[a]
                w, p, l = dists.get(a, (None, None, None))
                if w is not None:
                    dists[a] = ((1 - p_bj) * w, (1 - p_bj) * p, (1 - p_bj) * l + p_bj)

        ranked = sorted(evs.items(), key=lambda kv: kv[1], reverse=True)
        res["actions"] = [
            {"action": a, "ev": ev, **({"win": dists[a][0], "push": dists[a][1], "loss": dists[a][2]} if a in dists else {})}
            for a, ev in ranked
        ]
        res["best"] = ranked[0][0]

        if compare_basic:
            res["basic_strategy"] = self._basic_action(player, up, "ml" if use_ml else "exact")
        res["explanation"] = self._explain(res, counts, player, up, p_bj)
        if use_ml:
            feats = build_features(player, up, counts, rules)
            res["ml_outcome"] = PREDICTOR.predict_outcome(feats)
        return self._finish(res, t0)

    @staticmethod
    def _finish(res: dict, t0: float) -> dict:
        res["elapsed_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        return res

    def _dealer_probs(self, counts, up) -> dict:
        return dict(zip(OUTCOMES, self.dealer_distribution(counts, up, [[0] * 10])[:, 0].tolist()))

    def _solve_ml(self, counts, player, up) -> dict:
        """Stand y dealer exactos (una sola composición, ~1 ms); Hit/Double/Split por el modelo."""
        dealer = self.dealer_distribution(counts, up, [[0] * 10])
        total, _ = hand_total(player)
        win, push, ev = self._stand_outcomes(np.array([total]), dealer)
        actions = ["Hit"]
        if len(player) == 2:
            actions.append("Double")
            if player[0] == player[1]:
                actions.append("Split")
        feats = build_features(player, up, counts, self.rules)
        evs = {"Stand": float(ev[0]), **PREDICTOR.predict_evs(feats, actions)}
        dists = {"Stand": (float(win[0]), float(push[0]), float(1 - win[0] - push[0]))}
        return {"evs": evs, "dists": dists, "dealer": dealer[:, 0].tolist()}

    def _basic_action(self, player, up, source: str) -> str | None:
        """Mejor acción con un zapato neutro (sólo se retiran tu mano y la carta visible)."""
        deck = DeckState(self.rules.num_decks)
        try:
            deck.remove_many(list(player) + [up])
        except ValueError:
            return None
        counts, player = deck.as_tuple(), tuple(player)
        if source == "ml":
            evs = self._solve_ml(counts, player, up)["evs"]
        else:
            evs = self.solve_exact(counts, player, up)["evs"]
        evs = dict(evs)
        if self.rules.surrender and len(player) == 2:
            evs["Surrender"] = -0.5
        p_bj = self.dealer_blackjack_prob(counts, up)
        if self.rules.enhc and p_bj > 0:
            evs = {a: (1 - p_bj) * v - p_bj * _STAKE[a] for a, v in evs.items()}
        return max(evs, key=evs.get)

    # ---------------------------- Explicabilidad --------------------------- #
    def _explain(self, res: dict, counts, player, up, p_bj: float) -> str:
        actions = res["actions"]
        best = actions[0]
        second = actions[1] if len(actions) > 1 else None
        a, ev = best["action"], best["ev"]
        total, soft = res["player"]["total"], res["player"]["soft"]
        hand = f"{'blando ' if soft else ''}{total}"
        up_lbl = RANK_LABELS[up]
        bust = res["dealer"]["probs"]["bust"]
        stand = next(x for x in actions if x["action"] == "Stand")
        hard = sum(r + 1 for r in player)
        n = sum(counts)
        # prob. de pasarse con la siguiente carta (0 si la mano es blanda)
        p_bust_hit = 0.0 if soft else sum(c for r, c in enumerate(counts) if hard + r + 1 > 21) / n

        name = ACTION_ES[a]
        if a == "Stand":
            if total < 17:
                why = (f"el crupier se pasa un {bust:.1%} de las veces mostrando un {up_lbl}, "
                       f"lo cual es más favorable que el riesgo del {p_bust_hit:.1%} de pasarte pidiendo carta")
            elif p_bust_hit > 0:
                why = (f"con {hand} ganas el {stand['win']:.1%} de las manos plantándote "
                       f"(el crupier se pasa un {bust:.1%} con un {up_lbl}), y pedir te pasaría un {p_bust_hit:.1%} de las veces")
            else:
                why = (f"con {hand} ganas el {stand['win']:.1%} de las manos plantándote; "
                       f"una carta más no mejora lo suficiente contra un {up_lbl}")
        elif a == "Hit":
            if total < 17:
                risk = (f"aunque pedir te hace pasarte un {p_bust_hit:.1%} de las veces, el resto de cartas mejora tu mano"
                        if p_bust_hit > 0 else "y con una carta más no te puedes pasar")
                why = f"plantado con {hand} sólo ganas si el crupier se pasa ({bust:.1%} con un {up_lbl}); {risk}"
            else:
                why = (f"plantado con {hand} el crupier te supera un {stand['loss']:.1%} de las veces con un {up_lbl}; "
                       f"pedir (riesgo de pasarte: {p_bust_hit:.1%}) mejora tu expectativa")
        elif a == "Double":
            w = f"ganas el {best['win']:.1%} de las veces con una sola carta" if "win" in best else "tu mano es fuerte"
            why = (f"con {hand} contra un {up_lbl} {w} y el crupier se pasa un {bust:.1%}, "
                   f"así que duplicar la apuesta rinde más que jugar la mano normal")
        elif a == "Split":
            pair = RANK_LABELS[player[0]]
            why = f"dos manos que empiezan con {pair} valen más que jugar un {hand} contra un {up_lbl}"
        elif a == "Surrender":
            why = (f"perder media apuesta es mejor que la expectativa de jugar {hand} contra un {up_lbl} "
                   f"(ganarías sólo un {stand['win']:.1%} plantándote)")
        else:
            why = ""
        text = f"{name} es óptimo (EV {ev:+.3f}) porque {why}."
        if second:
            text += f" {ACTION_ES[second['action']]} rinde EV {second['ev']:+.3f} ({second['ev'] - ev:+.3f})."
        if self.rules.enhc and p_bj > 0:
            text += (f" ENHC: incluye un {p_bj:.1%} de blackjack del crupier, que se lleva también lo doblado o separado.")
        basic = res.get("basic_strategy")
        if basic and basic != a:
            tc = build_features(player, up, counts, self.rules)["true_count"]
            text += (f" ⚠ Desviación de la estrategia básica: con un zapato neutro lo óptimo sería "
                     f"{ACTION_ES.get(basic, basic)}; la composición actual (TC {tc:+.1f}) cambia la decisión.")
        elif basic:
            text += " Coincide con la estrategia básica (zapato neutro)."
        return text

    # ---------------------------- Ventaja exacta --------------------------- #
    def round_ev(self, counts: Sequence[int]) -> float:
        """
        EV exacto de una ronda completa (antes de repartir) jugando de forma óptima
        cada mano: suma sobre carta visible y las dos cartas del jugador.
        Sin seguro. Coste: ~550 análisis exactos (unos segundos).
        """
        c = list(counts)
        n = sum(c)
        total = 0.0
        payout = self.rules.blackjack_payout
        for u in range(10):
            if not c[u]:
                continue
            pu = c[u] / n
            c[u] -= 1
            n1 = n - 1
            for i in range(10):
                if not c[i]:
                    continue
                for j in range(i, 10):
                    avail_j = c[j] - (1 if i == j else 0)
                    if avail_j <= 0:
                        continue
                    p = c[i] * avail_j / (n1 * (n1 - 1)) * (1 if i == j else 2)
                    c[i] -= 1
                    c[j] -= 1
                    counts_ij = tuple(c)
                    p_bj = self.dealer_blackjack_prob(counts_ij, u)
                    tot, _ = hand_total((i, j))
                    if tot == 21:
                        ev = payout * (1 - p_bj)
                    else:
                        core = self.solve_exact(counts_ij, (i, j), u)["evs"]
                        evs = dict(core)
                        if self.rules.surrender:
                            evs["Surrender"] = -0.5
                        if self.rules.enhc:
                            ev = max((1 - p_bj) * v - p_bj * _STAKE[a] for a, v in evs.items())
                        else:  # peek: con blackjack del crupier se pierde sólo la apuesta inicial
                            ev = (1 - p_bj) * max(evs.values()) - p_bj
                    total += pu * p * ev
                    c[i] += 1
                    c[j] += 1
            c[u] += 1
        return total


@lru_cache(maxsize=4096)
def _solve_exact_cached(rules: Rules, counts: tuple, player: tuple, up: int) -> dict:
    return BlackjackCalculator(rules)._solve_exact(counts, player, up)
