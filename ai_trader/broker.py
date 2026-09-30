"""Portfolio state plus paper and live execution."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .models import Fill, Position

log = logging.getLogger(__name__)


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Portfolio:
    cash: float
    start_cash: float
    positions: dict[str, Position] = field(default_factory=dict)
    realized_pnl: float = 0.0
    fees_paid: float = 0.0
    day: str = ""  # YYYY-MM-DD of the current trading day (UTC)
    day_start_equity: float = 0.0
    n_trades: int = 0

    def equity(self, prices: dict[str, float]) -> float:
        value = self.cash
        for sym, p in self.positions.items():
            value += p.qty * prices.get(sym, p.entry_price)
        return value

    def roll_day(self, ts: str, equity: float) -> None:
        day = ts[:10]
        if day != self.day:
            self.day = day
            self.day_start_equity = equity

    def daily_pnl_pct(self, equity: float) -> float:
        if not self.day_start_equity:
            return 0.0
        return equity / self.day_start_equity - 1.0

    # --- persistence ---
    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "positions"}
        d["positions"] = {k: p.to_dict() for k, p in self.positions.items()}
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Portfolio":
        positions = {k: Position(**v) for k, v in d.pop("positions", {}).items()}
        return cls(positions=positions, **d)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load_or_create(cls, path: Path, start_cash: float) -> "Portfolio":
        if path.exists():
            return cls.from_dict(json.loads(path.read_text()))
        return cls(cash=start_cash, start_cash=start_cash)


class PaperBroker:
    """Simulated fills at the given price with fee and slippage."""

    name = "paper"

    def __init__(self, portfolio: Portfolio, fee_pct: float = 0.001, slippage_pct: float = 0.0005):
        self.p = portfolio
        self.fee_pct = fee_pct
        self.slippage_pct = slippage_pct

    def buy(self, symbol: str, notional: float, price: float, ts: str, stop_loss=None, take_profit=None, reason: str = "") -> Fill:
        fill_price = price * (1 + self.slippage_pct)
        fee = notional * self.fee_pct
        total = notional + fee
        if total > self.p.cash + 1e-9:
            raise ValueError(f"insufficient cash: need {total:.2f}, have {self.p.cash:.2f}")
        qty = notional / fill_price
        self.p.cash -= total
        self.p.fees_paid += fee
        self.p.positions[symbol] = Position(symbol, qty, fill_price, ts, stop_loss, take_profit, entry_fee=fee)
        self.p.n_trades += 1
        return Fill(ts, symbol, "buy", qty, fill_price, fee, 0.0, reason)

    def sell(self, symbol: str, qty: float, price: float, ts: str, reason: str = "") -> Fill:
        pos = self.p.positions[symbol]
        qty = min(qty, pos.qty)
        fill_price = price * (1 - self.slippage_pct)
        proceeds = qty * fill_price
        fee = proceeds * self.fee_pct
        entry_fee_part = pos.entry_fee * (qty / pos.qty)
        pnl = (fill_price - pos.entry_price) * qty - fee - entry_fee_part
        self.p.cash += proceeds - fee
        self.p.fees_paid += fee
        self.p.realized_pnl += pnl
        self.p.n_trades += 1
        if qty >= pos.qty - 1e-12:
            del self.p.positions[symbol]
        else:
            pos.qty -= qty
            pos.entry_fee -= entry_fee_part
        return Fill(ts, symbol, "sell", qty, fill_price, fee, pnl, reason)


class LiveBroker(PaperBroker):
    """Real market orders via ccxt. Mirrors fills into the local Portfolio.

    Only constructed when Settings.live_enabled is True. Stocks are not supported live.
    """

    name = "live"

    def __init__(self, portfolio: Portfolio, exchange, fee_pct: float = 0.001):
        super().__init__(portfolio, fee_pct=fee_pct, slippage_pct=0.0)
        self.ex = exchange

    @staticmethod
    def _avg(order: dict, fallback: float) -> float:
        return float(order.get("average") or order.get("price") or fallback)

    @staticmethod
    def _fee(order: dict, fallback: float) -> float:
        fee = order.get("fee") or {}
        return float(fee.get("cost") or fallback)

    def buy(self, symbol: str, notional: float, price: float, ts: str, stop_loss=None, take_profit=None, reason: str = "") -> Fill:
        qty = float(self.ex.amount_to_precision(symbol, notional / price))
        order = self.ex.create_market_buy_order(symbol, qty)
        order = self.ex.fetch_order(order["id"], symbol) if order.get("status") != "closed" else order
        fill_price = self._avg(order, price)
        filled = float(order.get("filled") or qty)
        fee = self._fee(order, filled * fill_price * self.fee_pct)
        log.warning("LIVE BUY %s qty=%s @ %s fee=%s", symbol, filled, fill_price, fee)
        self.p.cash -= filled * fill_price + fee
        self.p.fees_paid += fee
        self.p.positions[symbol] = Position(symbol, filled, fill_price, ts, stop_loss, take_profit, entry_fee=fee)
        self.p.n_trades += 1
        return Fill(ts, symbol, "buy", filled, fill_price, fee, 0.0, reason)

    def sell(self, symbol: str, qty: float, price: float, ts: str, reason: str = "") -> Fill:
        pos = self.p.positions[symbol]
        qty = float(self.ex.amount_to_precision(symbol, min(qty, pos.qty)))
        order = self.ex.create_market_sell_order(symbol, qty)
        order = self.ex.fetch_order(order["id"], symbol) if order.get("status") != "closed" else order
        fill_price = self._avg(order, price)
        filled = float(order.get("filled") or qty)
        fee = self._fee(order, filled * fill_price * self.fee_pct)
        entry_fee_part = pos.entry_fee * (filled / pos.qty)
        pnl = (fill_price - pos.entry_price) * filled - fee - entry_fee_part
        log.warning("LIVE SELL %s qty=%s @ %s pnl=%s", symbol, filled, fill_price, pnl)
        self.p.cash += filled * fill_price - fee
        self.p.fees_paid += fee
        self.p.realized_pnl += pnl
        self.p.n_trades += 1
        if filled >= pos.qty * 0.999:
            del self.p.positions[symbol]
        else:
            pos.qty -= filled
            pos.entry_fee -= entry_fee_part
        return Fill(ts, symbol, "sell", filled, fill_price, fee, pnl, reason)
