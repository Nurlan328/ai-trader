"""SQLite trade journal: every decision (approved or not) and every fill."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from .models import Decision, Fill

SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    symbol TEXT NOT NULL,
    price REAL NOT NULL,
    strategy TEXT NOT NULL,
    action TEXT NOT NULL,
    confidence REAL NOT NULL,
    size_pct REAL NOT NULL,
    approved INTEGER NOT NULL,
    risk_reason TEXT NOT NULL,
    reasoning TEXT NOT NULL,
    key_risks TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS fills (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    qty REAL NOT NULL,
    price REAL NOT NULL,
    fee REAL NOT NULL,
    pnl REAL NOT NULL,
    reason TEXT NOT NULL
);
"""


class Journal:
    def __init__(self, path: Path | str = ":memory:"):
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path))
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)

    def log_decision(self, ts: str, symbol: str, price: float, strategy: str, d: Decision, approved: bool, risk_reason: str) -> None:
        self.conn.execute(
            "INSERT INTO decisions (ts, symbol, price, strategy, action, confidence, size_pct, approved, risk_reason, reasoning, key_risks)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (ts, symbol, price, strategy, d.action, d.confidence, d.size_pct, int(approved), risk_reason, d.reasoning, json.dumps(d.key_risks)),
        )
        self.conn.commit()

    def log_fill(self, f: Fill) -> None:
        self.conn.execute(
            "INSERT INTO fills (ts, symbol, side, qty, price, fee, pnl, reason) VALUES (?,?,?,?,?,?,?,?)",
            (f.ts, f.symbol, f.side, f.qty, f.price, f.fee, f.pnl, f.reason),
        )
        self.conn.commit()

    def recent_decisions(self, n: int = 20) -> list[dict]:
        rows = self.conn.execute("SELECT * FROM decisions ORDER BY id DESC LIMIT ?", (n,)).fetchall()
        return [dict(r) for r in rows]

    def recent_fills(self, n: int = 20) -> list[dict]:
        rows = self.conn.execute("SELECT * FROM fills ORDER BY id DESC LIMIT ?", (n,)).fetchall()
        return [dict(r) for r in rows]

    def close(self) -> None:
        self.conn.close()
