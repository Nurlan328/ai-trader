"""Hard limits that sit between any strategy (including Claude) and the broker."""
from __future__ import annotations

from dataclasses import dataclass

from .broker import Portfolio
from .config import Settings
from .models import Decision


@dataclass
class Verdict:
    decision: Decision
    approved: bool
    reason: str
    notional: float = 0.0  # buy: quote currency to spend
    qty: float = 0.0  # sell: base quantity to close
    stop_loss: float | None = None
    take_profit: float | None = None


class RiskManager:
    def __init__(self, s: Settings):
        self.s = s

    def evaluate(self, d: Decision, symbol: str, price: float, portfolio: Portfolio, equity: float, atr_pct: float | None = None) -> Verdict:
        d = d.normalized()
        s = self.s
        if d.action == "hold":
            return Verdict(d, False, "hold")
        if d.confidence < s.min_confidence:
            return Verdict(d, False, f"confidence {d.confidence:.2f} < min {s.min_confidence:.2f}")

        pos = portfolio.positions.get(symbol)

        if d.action == "buy":
            if pos is not None:
                return Verdict(d, False, "already in position (no pyramiding)")
            if len(portfolio.positions) >= s.max_open_positions:
                return Verdict(d, False, f"max open positions ({s.max_open_positions}) reached")
            daily = portfolio.daily_pnl_pct(equity)
            if daily <= -s.max_daily_loss_pct:
                return Verdict(d, False, f"daily loss limit hit ({daily:.2%})")
            if d.size_pct <= 0:
                return Verdict(d, False, "size_pct is 0")
            cap = equity * s.max_position_pct * d.size_pct
            spendable = portfolio.cash / (1 + s.fee_pct + s.slippage_pct)
            notional = min(cap, spendable)
            if notional < s.min_notional:
                return Verdict(d, False, f"notional {notional:.2f} below minimum {s.min_notional}")
            sl_pct = d.stop_loss_pct or max(s.default_stop_loss_pct, 1.5 * (atr_pct or 0))
            tp_pct = d.take_profit_pct or max(s.default_take_profit_pct, 2 * sl_pct)
            if sl_pct <= 0:
                sl_pct = s.default_stop_loss_pct
            return Verdict(
                d, True, "approved buy", notional=notional,
                stop_loss=price * (1 - sl_pct), take_profit=price * (1 + tp_pct),
            )

        # sell
        if pos is None:
            return Verdict(d, False, "no position to sell")
        frac = d.size_pct if d.size_pct > 0 else 1.0
        qty = pos.qty * frac
        remaining = (pos.qty - qty) * price
        if remaining < s.min_notional:  # do not leave dust behind
            qty = pos.qty
        return Verdict(d, True, "approved sell", qty=qty)
