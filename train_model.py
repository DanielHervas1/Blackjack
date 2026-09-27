"""
Pipeline de Machine Learning del calculador de Blackjack.

Entrena dos modelos y los guarda juntos en `blackjack_model.pkl`:

A) Clasificador de resultado (Win / Push / Loss)
   Datos: dataset de Kaggle "900,000 hands of blackjack results" (+ history.db).
   El dataset registra el resultado final de manos jugadas con una estrategia fija
   y SIN composición del zapato, así que este modelo sólo usa features de la mano
   (total, blanda, pareja, carta del crupier). Da una probabilidad "empírica" de
   ganar, útil como referencia, pero no puede decidir entre acciones.

B) Modelo sustituto de EV (el que usa la API para baja latencia)
   Para predecir el EV de *cada acción* hacen falta etiquetas por acción, que el
   dataset no tiene. Se construye así:
     1. Se toman manos reales del dataset (2 cartas, o 3 si la mano pidió).
     2. Feature engineering: se simula un estado del zapato (nº de barajas,
        penetración y sesgo aleatorios) -> true count y composición exacta.
     3. Se etiqueta cada muestra con los EV exactos del motor recursivo
        (Stand / Hit / Double / Split), en paralelo.
     4. Se entrena un HistGradientBoostingRegressor por acción y se "compila" a
        arrays de NumPy para inferencia en < 1 ms.
   Además se añaden las manos guardadas en history.db calculadas en modo exacto.

Uso:
    python train_model.py                    # 60 000 muestras etiquetadas (~1-2 min)
    python train_model.py --samples 150000   # más precisión, más tiempo
    python train_model.py --no-kaggle       # manos sintéticas si no hay acceso a Kaggle
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from datetime import datetime, timezone
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np

from engine import MODEL_PATH, BlackjackCalculator, CompiledForest, Rules, hand_total
from features import FEATURE_NAMES, OUTCOME_FEATURES, build_features, to_vector

KAGGLE_DATASET = "mojocolors/900000-hands-of-blackjack-results"
ACTIONS = ("Stand", "Hit", "Double", "Split")
DECK_CHOICES = (1, 2, 6, 6, 6, 8)


# --------------------------------------------------------------------------- #
# 1. Datos
# --------------------------------------------------------------------------- #
def _value_to_index(v: int) -> int:
    """Valor del dataset (1 u 11 = As, 2..10) -> índice de rango (0 = As ... 9 = T)."""
    return 0 if v in (1, 11) else int(v) - 1


def load_kaggle() -> "pd.DataFrame":
    import kagglehub
    import pandas as pd

    path = kagglehub.dataset_download(KAGGLE_DATASET)
    files = glob.glob(os.path.join(path, "**", "*.csv"), recursive=True)
    if not files:
        raise FileNotFoundError(f"No hay CSV en {path}")
    df = pd.read_csv(files[0])
    print(f"[datos] {len(df):,} manos de {files[0]}")

    cards = df[["card1", "card2", "card3", "card4", "card5"]].to_numpy()
    out = pd.DataFrame({
        # cartas del jugador en orden, ignorando huecos (0)
        "player": [tuple(_value_to_index(v) for v in row if v) for row in cards],
        "up": [_value_to_index(v) for v in df["dealcard1"]],
        "outcome": df["winloss"].astype(str),
    })
    return out


def synthetic_hands(n: int, rng: np.random.Generator) -> "pd.DataFrame":
    """Alternativa sin Kaggle: manos aleatorias de un zapato infinito (sin resultado)."""
    import pandas as pd

    p = np.array([1] * 9 + [4]) / 13
    draw = lambda k: tuple(int(x) for x in rng.choice(10, size=k, p=p))  # noqa: E731
    return pd.DataFrame({"player": [draw(3) for _ in range(n)],
                         "up": [int(x) for x in rng.choice(10, size=n, p=p)],
                         "outcome": None})


# --------------------------------------------------------------------------- #
# 2. Feature engineering: simular el estado del zapato
# --------------------------------------------------------------------------- #
def simulate_shoe(num_decks: int, rng: np.random.Generator) -> np.ndarray:
    """
    Zapato parcialmente jugado: penetración uniforme 0-85 % y, en la mitad de los
    casos, un sesgo hacia cartas altas o bajas para cubrir true counts extremos.
    """
    initial = np.array([4 * num_decks] * 9 + [16 * num_decks])
    n = int(initial.sum())
    k = int(rng.uniform(0, 0.85) * n)
    if k == 0:
        return initial
    if rng.random() < 0.5:
        removed = rng.multivariate_hypergeometric(initial, k)
    else:
        tilt = rng.uniform(-1.5, 1.5)  # >0 retira más cartas bajas (TC positivo)
        hilo = np.array([-1, 1, 1, 1, 1, 1, 0, 0, 0, -1])
        weights = np.exp(tilt * hilo)
        cards = np.repeat(np.arange(10), initial)
        probs = weights[cards] / weights[cards].sum()
        idx = rng.choice(n, size=k, replace=False, p=probs)
        removed = np.bincount(cards[idx], minlength=10)
    return initial - removed


def make_samples(hands, n: int, rng: np.random.Generator) -> list[tuple]:
    """(counts, player, up, rules_dict) listos para etiquetar."""
    tasks = []
    rows = hands.sample(n=min(n * 2, len(hands)), random_state=int(rng.integers(1 << 31)), replace=n * 2 > len(hands))
    for player, up in zip(rows["player"], rows["up"]):
        if len(tasks) >= n:
            break
        hand = list(player[:2])
        # 30 %: estado tras pedir una carta (si la mano real pidió y no se pasó)
        if len(player) >= 3 and rng.random() < 0.3 and hand_total(player[:3])[0] <= 21:
            hand = list(player[:3])
        total, _ = hand_total(hand)
        if len(hand) == 2 and total == 21:  # blackjack: no hay decisión
            continue
        rules = {"num_decks": int(rng.choice(DECK_CHOICES)),
                 "dealer_hits_soft17": bool(rng.random() < 0.5),
                 "double_after_split": bool(rng.random() < 0.7)}
        counts = simulate_shoe(rules["num_decks"], rng)
        for c in hand + [up]:
            counts[c] -= 1
        if (counts < 0).any() or counts.sum() < 20:
            continue
        tasks.append((tuple(int(x) for x in counts), tuple(hand), int(up), rules))
    return tasks


def label(task: tuple) -> tuple[dict, dict]:
    """Etiqueta exacta (EV condicionado a que el crupier no tenga blackjack)."""
    counts, player, up, rules = task
    r = Rules(**rules)
    evs = BlackjackCalculator(r).solve_exact(counts, player, up)["evs"]
    return build_features(player, up, counts, r), dict(evs)


# --------------------------------------------------------------------------- #
# 3. Historial SQLite
# --------------------------------------------------------------------------- #
def history_rows() -> list[dict]:
    try:
        import history
    except ImportError:
        return []
    return history.load_training_rows()


def history_ev_samples(rows: list[dict]) -> list[tuple[dict, dict]]:
    """Manos del historial calculadas en modo exacto -> muestras (features, EVs)."""
    out = []
    for r in rows:
        if r.get("engine_source") != "exact" or any(r.get(f) is None for f in FEATURE_NAMES):
            continue
        feats = {f: r[f] for f in FEATURE_NAMES}
        evs = {a: r[f"ev_{a.lower()}"] for a in ACTIONS if r.get(f"ev_{a.lower()}") is not None}
        # los EV guardados con ENHC incluyen el blackjack del crupier: se deshace el ajuste
        if r.get("enhc"):
            up = r["dealer_upcard"]
            p = r["frac_T"] if up == 11 else r["frac_A"] if up == 10 else 0.0
            stake = {"Stand": 1, "Hit": 1, "Double": 2, "Split": 2}
            if p < 1:
                evs = {a: (v + p * stake[a]) / (1 - p) for a, v in evs.items()}
        out.append((feats, evs))
    return out


# --------------------------------------------------------------------------- #
# 4. Entrenamiento
# --------------------------------------------------------------------------- #
def train_outcome_model(hands, hist_rows: list[dict], rng: np.random.Generator, max_rows: int):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import accuracy_score, log_loss
    from sklearn.model_selection import train_test_split

    data = hands.dropna(subset=["outcome"])
    if len(data) > max_rows:
        data = data.sample(n=max_rows, random_state=int(rng.integers(1 << 31)))
    X, y = [], []
    dummy_rules = Rules()
    blank = [4] * 9 + [16]
    for player, up, outcome in zip(data["player"], data["up"], data["outcome"]):
        f = build_features(player[:2], up, blank, dummy_rules)
        X.append(to_vector(f, OUTCOME_FEATURES))
        y.append(outcome)
    result_map = {"win": "Win", "blackjack": "Win", "push": "Push", "loss": "Loss", "surrender": "Loss"}
    for r in hist_rows:
        if r.get("result") in result_map and all(r.get(f) is not None for f in OUTCOME_FEATURES):
            X.append([float(r[f]) for f in OUTCOME_FEATURES])
            y.append(result_map[r["result"]])
    if not X:
        return None, {}
    X, y = np.array(X), np.array(y)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.15, random_state=0, stratify=y)
    clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=0)
    clf.fit(Xtr, ytr)
    proba = clf.predict_proba(Xte)
    prior = np.tile([np.mean(ytr == c) for c in clf.classes_], (len(yte), 1))
    metrics = {
        "rows": int(len(X)),
        "accuracy": float(accuracy_score(yte, clf.predict(Xte))),
        "log_loss": float(log_loss(yte, proba, labels=clf.classes_)),
        "log_loss_baseline": float(log_loss(yte, prior, labels=clf.classes_)),
    }
    return clf, metrics


def outcome_table(clf) -> dict:
    """
    Las 4 features del clasificador son discretas: se precalculan todas las
    combinaciones para que la API responda con una búsqueda en diccionario en
    lugar de llamar a predict_proba (varios ms por fila).
    """
    keys = [(t, s, p, u) for t in range(4, 22) for s in (0, 1) for p in (0, 1) for u in range(2, 12)]
    proba = clf.predict_proba(np.array(keys, dtype=float))
    return {k: {str(c): float(v) for c, v in zip(clf.classes_, row)} for k, row in zip(keys, proba)}


def train_ev_models(samples: list[tuple[dict, dict]]):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.metrics import mean_absolute_error
    from sklearn.model_selection import train_test_split

    X = np.array([to_vector(f) for f, _ in samples])
    idx_tr, idx_te = train_test_split(np.arange(len(samples)), test_size=0.15, random_state=0)
    models, metrics = {}, {}
    preds_te: dict[str, np.ndarray] = {}
    for a in ACTIONS:
        mask = np.array([a in evs and np.isfinite(evs[a]) for _, evs in samples])
        y = np.array([evs.get(a, np.nan) for _, evs in samples], dtype=float)
        tr = idx_tr[mask[idx_tr]]
        te = idx_te[mask[idx_te]]
        if len(tr) < 50:
            print(f"[ev] {a}: muy pocas muestras ({len(tr)}), se omite")
            continue
        model = HistGradientBoostingRegressor(max_iter=600, learning_rate=0.06, max_leaf_nodes=63,
                                              l2_regularization=1e-3, random_state=0)
        model.fit(X[tr], y[tr])
        models[a] = model
        if len(te):
            pred = model.predict(X[te])
            metrics[a] = {"train": int(len(tr)), "test": int(len(te)), "mae": float(mean_absolute_error(y[te], pred))}
            full = np.full(len(samples), np.nan)
            full[te] = pred
            preds_te[a] = full
        print(f"[ev] {a:6s} n={len(tr):6d}  MAE test={metrics.get(a, {}).get('mae', float('nan')):.4f}")

    # acierto de la decisión: ¿la mejor acción predicha coincide con la exacta?
    agree = regret = 0.0
    count = 0
    for i in idx_te:
        evs = samples[i][1]
        cand = [a for a in models if a in evs and np.isfinite(evs[a]) and not np.isnan(preds_te[a][i])]
        if len(cand) < 2:
            continue
        pred_best = max(cand, key=lambda a: preds_te[a][i])
        true_best = max(cand, key=lambda a: evs[a])
        agree += pred_best == true_best
        regret += evs[true_best] - evs[pred_best]
        count += 1
    if count:
        metrics["decision_agreement"] = agree / count
        metrics["mean_regret_ev"] = regret / count
        print(f"[ev] Acierto de la decisión óptima: {agree / count:.2%}   pérdida media de EV: {regret / count:.5f}")
    return models, metrics


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description="Entrena blackjack_model.pkl")
    ap.add_argument("--samples", type=int, default=60000, help="muestras etiquetadas con el motor exacto")
    ap.add_argument("--outcome-rows", type=int, default=300000, help="filas de Kaggle para el clasificador")
    ap.add_argument("--workers", type=int, default=max(1, cpu_count() - 1))
    ap.add_argument("--no-kaggle", action="store_true", help="no descargar Kaggle (manos sintéticas)")
    ap.add_argument("--no-history", action="store_true", help="no usar history.db")
    ap.add_argument("--out", type=Path, default=MODEL_PATH)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import joblib

    rng = np.random.default_rng(args.seed)
    t0 = time.time()

    if args.no_kaggle:
        hands = synthetic_hands(max(args.samples * 3, 10000), rng)
    else:
        try:
            hands = load_kaggle()
        except Exception as exc:
            print(f"[datos] No se pudo descargar Kaggle ({exc}). Usa --no-kaggle o configura ~/.kaggle/kaggle.json")
            return 1

    hist = [] if args.no_history else history_rows()
    print(f"[datos] {len(hist)} manos en history.db")

    # A) clasificador de resultado
    outcome_model, outcome_metrics = (None, {})
    if not args.no_kaggle or hist:
        print("[A] Entrenando clasificador Win/Push/Loss ...")
        outcome_model, outcome_metrics = train_outcome_model(hands, hist, rng, args.outcome_rows)
        if outcome_metrics:
            print(f"[A] accuracy={outcome_metrics['accuracy']:.3f}  log_loss={outcome_metrics['log_loss']:.4f}"
                  f"  (baseline {outcome_metrics['log_loss_baseline']:.4f})")

    # B) modelo sustituto de EV
    tasks = make_samples(hands, args.samples, rng)
    print(f"[B] Etiquetando {len(tasks):,} estados con el motor exacto ({args.workers} procesos) ...")
    t1 = time.time()
    with Pool(args.workers) as pool:
        samples = []
        for i, s in enumerate(pool.imap_unordered(label, tasks, chunksize=32), 1):
            samples.append(s)
            if i % 2000 == 0:
                print(f"    {i:,}/{len(tasks):,}  ({time.time() - t1:.0f}s)")
    samples += history_ev_samples(hist)
    print(f"[B] {len(samples):,} muestras en {time.time() - t1:.0f}s. Entrenando regresores ...")
    ev_models, ev_metrics = train_ev_models(samples)

    # árboles compilados a NumPy para inferencia en < 1 ms (sklearn.predict tarda ~5 ms/fila)
    compiled = {a: CompiledForest.arrays_from_hgb(m) for a, m in ev_models.items()}
    forests = [CompiledForest(arr) for arr in compiled.values()]
    x = np.array(to_vector(samples[0][0]))
    t2 = time.perf_counter()
    for _ in range(200):
        for f in forests:
            f.predict_one(x)
    latency_ms = (time.perf_counter() - t2) / 200 * 1000
    print(f"[B] Latencia de inferencia (todas las acciones): {latency_ms:.2f} ms")

    bundle = {
        "version": 1,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "feature_names": FEATURE_NAMES,
        "ev_models": ev_models,
        "compiled": compiled,
        "outcome_model": outcome_model,
        "outcome_table": outcome_table(outcome_model) if outcome_model is not None else None,
        "outcome_features": OUTCOME_FEATURES,
        "n_samples": len(samples),
        "metrics": {"ev": ev_metrics, "outcome": outcome_metrics, "latency_ms": latency_ms},
    }
    joblib.dump(bundle, args.out)
    print(f"Modelo guardado en {args.out} ({time.time() - t0:.0f}s en total)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
