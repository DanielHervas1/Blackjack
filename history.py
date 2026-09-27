"""
Persistencia del historial de manos en SQLite (librería estándar `sqlite3`).

Las columnas de features se generan a partir de `features.FEATURE_NAMES`, de modo
que cada fila de `hands` es directamente una muestra (X, y) para re-entrenar:
  X = FEATURE_NAMES
  y = ev_* (regresión del EV por acción), result / net_units (resultado real),
      action_taken vs recommended_action (calidad de la decisión).
"""
from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from features import FEATURE_NAMES

DB_PATH = Path(os.environ.get("BJ_DB_PATH", Path(__file__).parent / "history.db"))

EV_COLUMNS = ["ev_stand", "ev_hit", "ev_double", "ev_split", "ev_surrender"]
META_COLUMNS = {
    "created_at": "TEXT NOT NULL",
    "player_cards": "TEXT NOT NULL",   # p.ej. "T,6" (mano en el momento de decidir)
    "final_cards": "TEXT",             # mano al terminar (tras pedir)
    "dealer_card": "TEXT NOT NULL",    # etiqueta de la carta visible
    "running_count": "INTEGER",
    "penetration": "REAL",
    "composition": "TEXT",             # restantes por valor "A:24,2:23,..."
    "blackjack_payout": "REAL",
    "surrender_allowed": "INTEGER",
    "advantage": "REAL",               # ventaja estimada antes de la mano
    "bet": "REAL",
    "recommended_action": "TEXT",
    "basic_action": "TEXT",
    "action_taken": "TEXT",
    "result": "TEXT",                  # win | push | loss | blackjack | surrender
    "net_units": "REAL",               # ganancia neta en unidades de la apuesta inicial
    "engine_source": "TEXT",           # exact | ml
}
RESULTS = ("win", "push", "loss", "blackjack", "surrender")


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    cols = [f"{k} {v}" for k, v in META_COLUMNS.items()]
    cols += [f"{f} REAL" for f in FEATURE_NAMES]
    cols += [f"{c} REAL" for c in EV_COLUMNS]
    with _connect() as conn:
        conn.execute(f"CREATE TABLE IF NOT EXISTS hands (id INTEGER PRIMARY KEY AUTOINCREMENT, {', '.join(cols)})")
        # migración suave: añadir columnas nuevas si el esquema de features creció
        existing = {r["name"] for r in conn.execute("PRAGMA table_info(hands)")}
        for col in cols:
            name = col.split()[0]
            if name not in existing:
                conn.execute(f"ALTER TABLE hands ADD COLUMN {col}")


def save_hand(row: dict) -> int:
    row = {**row, "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    allowed = set(META_COLUMNS) | set(FEATURE_NAMES) | set(EV_COLUMNS)
    data = {k: v for k, v in row.items() if k in allowed}
    keys = list(data)
    with _connect() as conn:
        cur = conn.execute(
            f"INSERT INTO hands ({', '.join(keys)}) VALUES ({', '.join('?' for _ in keys)})",
            [data[k] for k in keys],
        )
        return int(cur.lastrowid)


def recent(limit: int = 25) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute("SELECT * FROM hands ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in rows]


def stats() -> dict:
    with _connect() as conn:
        r = conn.execute(
            """SELECT COUNT(*) AS hands,
                      COALESCE(SUM(net_units), 0) AS net_units,
                      COALESCE(SUM(net_units * bet), 0) AS net_money,
                      COALESCE(SUM(CASE WHEN result IN ('win','blackjack') THEN 1 ELSE 0 END), 0) AS wins,
                      COALESCE(SUM(CASE WHEN result = 'push' THEN 1 ELSE 0 END), 0) AS pushes,
                      COALESCE(SUM(CASE WHEN action_taken = recommended_action THEN 1 ELSE 0 END), 0) AS followed,
                      COALESCE(SUM(CASE WHEN action_taken = recommended_action THEN
                          0 ELSE 1 END), 0) AS deviations
               FROM hands"""
        ).fetchone()
    return dict(r)


def delete(hand_id: int) -> bool:
    with _connect() as conn:
        return conn.execute("DELETE FROM hands WHERE id = ?", (hand_id,)).rowcount > 0


def load_training_rows() -> list[dict]:
    """Filas completas para train_model.py (features + targets)."""
    if not DB_PATH.exists():
        return []
    with _connect() as conn:
        return [dict(r) for r in conn.execute("SELECT * FROM hands")]
