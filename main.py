"""
API FastAPI del calculador de EV de Blackjack.

El servidor guarda el estado del zapato (cartas ya descartadas en rondas anteriores).
Cada /calculate recibe las cartas de la ronda en curso (mi mano, carta del crupier y
cartas de mesa) y las descuenta del zapato antes de calcular.

Arranque:  uvicorn main:app --reload
"""
from __future__ import annotations

import threading
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import history
from engine import (PREDICTOR, BlackjackCalculator, DeckState, Rules, estimate_advantage,
                    kelly_bet, rank_index)
from features import build_features

STATIC_DIR = Path(__file__).parent / "static"


class Shoe:
    """Estado del zapato en memoria (aplicación local de un solo usuario)."""

    def __init__(self) -> None:
        self.lock = Lock()
        self.rules = Rules()
        self.deck = DeckState(self.rules.num_decks)
        self.rounds: list[list[str]] = []  # rondas descartadas, para poder deshacer
        # EV exacto de una ronda con zapato neutro, por reglas (se calcula en segundo plano)
        self.base_edges: dict[Rules, float] = {}
        self.exact_cache: dict[tuple, float] = {}

    def state(self) -> dict:
        return {
            "rules": self.rules.to_dict(),
            "shoe": self.deck.summary(),
            "rounds": len(self.rounds),
            "model": PREDICTOR.status(),
            "exact_base_edge": self.base_edges.get(self.rules),
        }


shoe = Shoe()


def _warm_up(rules: Rules) -> None:
    """
    Calcula en segundo plano el EV exacto de una ronda con zapato neutro (~3 s).
    Sirve de ventaja base exacta para el estimador de Kelly y, de paso, precalienta
    la caché de la estrategia básica (las 550 combinaciones mano/carta visible).
    """
    if rules in shoe.base_edges:
        return
    ev = BlackjackCalculator(rules).round_ev(DeckState(rules.num_decks).counts)
    shoe.base_edges[rules] = ev


def _start_warm_up(rules: Rules) -> None:
    threading.Thread(target=_warm_up, args=(rules,), daemon=True).start()


@asynccontextmanager
async def lifespan(_: FastAPI):
    history.init_db()
    PREDICTOR.bundle()  # carga el modelo (si existe) antes de la primera petición
    _start_warm_up(shoe.rules)
    yield


app = FastAPI(title="Blackjack EV — Exact Deck Composition", lifespan=lifespan)


# --------------------------------------------------------------------------- #
# Modelos
# --------------------------------------------------------------------------- #
Mode = Literal["auto", "exact", "ml"]


class ResetRequest(BaseModel):
    num_decks: int = Field(6, ge=1, le=8)
    dealer_hits_soft17: bool = False
    double_after_split: bool = True
    blackjack_payout: float = Field(1.5, gt=0, le=3)
    surrender: bool = False
    enhc: bool = False


class CardsRequest(BaseModel):
    cards: list[str] = Field(default_factory=list)


class RoundCards(BaseModel):
    player_cards: list[str] = Field(default_factory=list)
    dealer_upcard: Optional[str] = None
    dead_cards: list[str] = Field(default_factory=list)


class BetSettings(BaseModel):
    bankroll: float = Field(0, ge=0)
    kelly_fraction: float = Field(0.5, gt=0, le=1)
    min_bet: float = Field(10, ge=0)
    max_bet: Optional[float] = Field(None, ge=0)


class CalculateRequest(RoundCards):
    mode: Mode = "auto"
    bet: Optional[BetSettings] = None


class SaveHandRequest(BaseModel):
    decision: RoundCards  # estado en el momento de decidir (mano de 2 cartas)
    round_cards: list[str] = Field(default_factory=list)  # todas las cartas vistas en la ronda
    final_cards: list[str] = Field(default_factory=list)
    action_taken: str
    result: Literal["win", "push", "loss", "blackjack", "surrender"]
    bet: Optional[float] = Field(None, ge=0)
    net_units: Optional[float] = None
    mode: Mode = "auto"


def _bad_request(exc: Exception) -> HTTPException:
    return HTTPException(status_code=400, detail=str(exc))


def _round_deck(base: DeckState, cards: RoundCards) -> tuple[DeckState, list[int], Optional[int]]:
    """Zapato menos las cartas de la ronda; devuelve (zapato, mano, carta visible)."""
    deck = base.copy()
    player = [rank_index(c) for c in cards.player_cards]
    up = rank_index(cards.dealer_upcard) if cards.dealer_upcard else None
    deck.remove_many(cards.dead_cards)
    deck.remove_many(player)
    if up is not None:
        deck.remove(up)
    return deck, player, up


def _advantage(counts, rules: Rules) -> dict:
    adv = estimate_advantage(counts, rules)
    exact_base = shoe.base_edges.get(rules)
    if exact_base is not None:  # base exacta del motor + desplazamiento por composición
        adv = {**adv, "advantage": exact_base + adv["shift"], "base_edge": exact_base, "base_source": "exact"}
    else:
        adv = {**adv, "base_source": "tabla"}
    return adv


# --------------------------------------------------------------------------- #
# Endpoints: zapato
# --------------------------------------------------------------------------- #
@app.get("/state")
def get_state() -> dict:
    with shoe.lock:
        return shoe.state()


@app.post("/reset")
def reset(req: ResetRequest | None = None) -> dict:
    """Baraja de nuevo: zapato completo con las reglas indicadas."""
    req = req or ResetRequest()
    with shoe.lock:
        shoe.rules = Rules(**req.model_dump())
        shoe.deck = DeckState(req.num_decks)
        shoe.rounds.clear()
        _start_warm_up(shoe.rules)
        return shoe.state()


@app.post("/discard")
def discard(req: CardsRequest) -> dict:
    """Fin de ronda: retira definitivamente del zapato todas las cartas vistas."""
    with shoe.lock:
        _commit(req.cards)
        return shoe.state()


def _commit(cards: list[str]) -> None:
    deck = shoe.deck.copy()
    try:
        deck.remove_many(cards)
    except ValueError as exc:
        raise _bad_request(exc)
    shoe.deck = deck
    shoe.rounds.append([str(c).upper() for c in cards])


@app.post("/undo-round")
def undo_round() -> dict:
    """Devuelve al zapato las cartas de la última ronda descartada."""
    with shoe.lock:
        if not shoe.rounds:
            raise HTTPException(status_code=400, detail="No hay rondas que deshacer")
        cards = shoe.rounds.pop()
        for c in cards:
            shoe.deck.add(c)
        return {**shoe.state(), "restored": cards}


# --------------------------------------------------------------------------- #
# Endpoints: cálculo
# --------------------------------------------------------------------------- #
@app.post("/calculate")
def calculate(req: CalculateRequest) -> dict:
    """EV de cada acción con la composición actual del zapato, explicación, ventaja y Kelly."""
    with shoe.lock:
        rules, base = shoe.rules, shoe.deck.copy()
    try:
        deck, player, up = _round_deck(base, req)
    except ValueError as exc:
        raise _bad_request(exc)

    result = BlackjackCalculator(rules).analyze(deck.counts, player, up, mode=req.mode)
    result["shoe"] = deck.summary()
    result["rules"] = rules.to_dict()
    result["advantage"] = _advantage(deck.counts, rules)
    if req.bet and req.bet.bankroll > 0:
        b = req.bet
        result["kelly"] = kelly_bet(result["advantage"]["advantage"], b.bankroll, b.kelly_fraction, b.min_bet, b.max_bet)
    result["model_available"] = PREDICTOR.available
    return result


@app.post("/advantage/exact")
def advantage_exact(req: RoundCards) -> dict:
    """EV exacto de la próxima ronda con la composición actual (tarda unos segundos)."""
    with shoe.lock:
        rules, base = shoe.rules, shoe.deck.copy()
    try:
        deck, _, _ = _round_deck(base, req)
    except ValueError as exc:
        raise _bad_request(exc)
    key = (rules, deck.as_tuple())
    if key not in shoe.exact_cache:
        shoe.exact_cache[key] = BlackjackCalculator(rules).round_ev(deck.counts)
    return {"advantage": shoe.exact_cache[key], "estimate": _advantage(deck.counts, rules)}


@app.get("/model/status")
def model_status() -> dict:
    return PREDICTOR.status()


# --------------------------------------------------------------------------- #
# Endpoints: historial (SQLite)
# --------------------------------------------------------------------------- #
def _net_units(action: str, result: str, payout: float) -> float:
    stake = 2.0 if action in ("Double", "Split") else 1.0
    return {"win": stake, "loss": -stake, "push": 0.0, "blackjack": payout, "surrender": -0.5}[result]


@app.post("/history/save")
def history_save(req: SaveHandRequest) -> dict:
    """Guarda la mano (features + EVs + acción + resultado) y retira la ronda del zapato."""
    with shoe.lock:
        rules, base = shoe.rules, shoe.deck.copy()
        try:
            deck, player, up = _round_deck(base, req.decision)
            base.copy().remove_many(req.round_cards)  # valida antes de guardar nada
        except ValueError as exc:
            raise _bad_request(exc)
        if up is None or len(player) < 2:
            raise HTTPException(status_code=400, detail="Hace falta tu mano (2+ cartas) y la carta del crupier")

        analysis = BlackjackCalculator(rules).analyze(deck.counts, player, up, mode=req.mode)
        evs = {a["action"]: a["ev"] for a in analysis["actions"]}
        summary = deck.summary()
        adv = _advantage(deck.counts, rules)
        row = {
            **build_features(player, up, deck.counts, rules),
            "player_cards": ",".join(req.decision.player_cards).upper(),
            "final_cards": ",".join(req.final_cards).upper() or None,
            "dealer_card": req.decision.dealer_upcard.upper(),
            "running_count": summary["running_count"],
            "penetration": summary["penetration"],
            "composition": ",".join(f"{k}:{v}" for k, v in summary["remaining"].items()),
            "blackjack_payout": rules.blackjack_payout,
            "surrender_allowed": int(rules.surrender),
            "advantage": adv["advantage"],
            "bet": req.bet,
            "recommended_action": analysis["best"],
            "basic_action": analysis.get("basic_strategy"),
            "action_taken": req.action_taken,
            "result": req.result,
            "net_units": req.net_units if req.net_units is not None
            else _net_units(req.action_taken, req.result, rules.blackjack_payout),
            "engine_source": analysis.get("source"),
            "ev_stand": evs.get("Stand"),
            "ev_hit": evs.get("Hit"),
            "ev_double": evs.get("Double"),
            "ev_split": evs.get("Split"),
            "ev_surrender": evs.get("Surrender"),
        }
        hand_id = history.save_hand(row)
        _commit(req.round_cards)
        return {**shoe.state(), "saved_id": hand_id}


@app.get("/history")
def history_list(limit: int = 25) -> dict:
    return {"hands": history.recent(max(1, min(limit, 500))), "stats": history.stats()}


@app.delete("/history/{hand_id}")
def history_delete(hand_id: int) -> dict:
    if not history.delete(hand_id):
        raise HTTPException(status_code=404, detail="Mano no encontrada")
    return {"deleted": hand_id}


# --------------------------------------------------------------------------- #
# Frontend
# --------------------------------------------------------------------------- #
@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
