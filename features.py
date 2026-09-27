"""
Definición única de las features (X) del modelo.

La usan tres sitios, así que el esquema es siempre el mismo:
  * engine.py      -> inferencia en vivo
  * history.py     -> columnas de la tabla SQLite (dataset de re-entrenamiento)
  * train_model.py -> entrenamiento
"""
from __future__ import annotations

from typing import Sequence

RANK_LABELS = ("A", "2", "3", "4", "5", "6", "7", "8", "9", "T")

FEATURE_NAMES: list[str] = [
    "player_total",     # total de la mano (el as cuenta 11 si no se pasa)
    "player_soft",      # 1 si la mano es blanda
    "is_pair",          # 1 si son dos cartas del mismo valor
    "pair_value",       # valor de la pareja (2..11), 0 si no hay pareja
    "num_cards",        # nº de cartas en la mano
    "dealer_upcard",    # valor de la carta visible del crupier (2..11, As = 11)
    "true_count",       # Hi-Lo running count / barajas restantes
    "decks_remaining",
    *[f"frac_{r}" for r in RANK_LABELS],  # composición exacta: fracción de cada valor en el zapato
    "num_decks",
    "h17",
    "das",
    "enhc",
]

# El clasificador de resultado se entrena con el dataset de Kaggle, que no incluye
# composición del zapato: sólo puede usar las features de la mano.
OUTCOME_FEATURES: list[str] = ["player_total", "player_soft", "is_pair", "dealer_upcard"]


def card_value(idx: int) -> int:
    """Índice de rango (0 = As ... 9 = T) -> valor de blackjack (As = 11)."""
    return 11 if idx == 0 else idx + 1


def build_features(player: Sequence[int], up: int, counts: Sequence[int], rules) -> dict[str, float]:
    """
    player / up: índices de rango. counts: zapato restante (sin las cartas visibles).
    rules: engine.Rules (u objeto con los mismos atributos).
    """
    hard = sum(r + 1 for r in player)
    soft = 0 in player and hard + 10 <= 21
    total = hard + 10 if soft else hard
    pair = len(player) == 2 and player[0] == player[1]

    n = sum(counts) or 1
    initial = [4 * rules.num_decks] * 9 + [16 * rules.num_decks]
    seen = [i - c for i, c in zip(initial, counts)]
    running = sum(seen[1:6]) - seen[0] - seen[9]
    decks_left = n / 52

    feats: dict[str, float] = {
        "player_total": total,
        "player_soft": int(soft),
        "is_pair": int(pair),
        "pair_value": card_value(player[0]) if pair else 0,
        "num_cards": len(player),
        "dealer_upcard": card_value(up),
        "true_count": running / decks_left if decks_left > 0 else 0.0,
        "decks_remaining": decks_left,
        "num_decks": rules.num_decks,
        "h17": int(rules.dealer_hits_soft17),
        "das": int(rules.double_after_split),
        "enhc": int(rules.enhc),
    }
    for lbl, c in zip(RANK_LABELS, counts):
        feats[f"frac_{lbl}"] = c / n
    return feats


def to_vector(feats: dict[str, float], names: Sequence[str] = FEATURE_NAMES) -> list[float]:
    return [float(feats[k]) for k in names]
