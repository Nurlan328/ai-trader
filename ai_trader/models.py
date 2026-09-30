from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

from pydantic import BaseModel, Field

Action = Literal["buy", "sell", "hold"]


class Decision(BaseModel):
    """What the strategy wants to do. Produced by Claude (structured output) or by rules.

    Numeric fields are validated/clamped in `normalized()` rather than with schema
    constraints so the JSON schema stays compatible with structured outputs.
    """

    action: Action = Field(description="buy = open a long, sell = close some or all of the position, hold = do nothing")
    confidence: float = Field(description="Probability 0..1 that the action is right; below 0.6 should be a hold")
    size_pct: float = Field(description="0..1. Buy: fraction of the max allowed position to deploy. Sell: fraction of the position to close")
    stop_loss_pct: float | None = Field(default=None, description="Buy only: stop distance below entry as a fraction, e.g. 0.03 = 3%")
    take_profit_pct: float | None = Field(default=None, description="Buy only: target distance above entry as a fraction, e.g. 0.06 = 6%")
    reasoning: str = Field(description="2-4 short sentences citing the specific numbers behind the decision")
    key_risks: list[str] = Field(default_factory=list, description="1-3 short strings: what would invalidate this idea")

    @classmethod
    def hold(cls, reason: str) -> "Decision":
        return cls(action="hold", confidence=0.0, size_pct=0.0, reasoning=reason)

    def normalized(self) -> "Decision":
        def clamp(v: float | None, lo: float, hi: float) -> float | None:
            if v is None:
                return None
            return max(lo, min(hi, float(v)))

        return self.model_copy(
            update={
                "confidence": clamp(self.confidence, 0.0, 1.0),
                "size_pct": clamp(self.size_pct, 0.0, 1.0),
                "stop_loss_pct": clamp(self.stop_loss_pct, 0.0, 0.5),
                "take_profit_pct": clamp(self.take_profit_pct, 0.0, 2.0),
            }
        )


@dataclass
class Position:
    symbol: str
    qty: float
    entry_price: float
    opened_at: str  # ISO-8601 UTC
    stop_loss: float | None = None
    take_profit: float | None = None
    entry_fee: float = 0.0  # paid on the way in; charged against P&L on the way out

    def unrealized_pnl(self, price: float) -> float:
        return (price - self.entry_price) * self.qty

    def unrealized_pct(self, price: float) -> float:
        return price / self.entry_price - 1.0 if self.entry_price else 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Fill:
    ts: str
    symbol: str
    side: Literal["buy", "sell"]
    qty: float
    price: float
    fee: float
    pnl: float = 0.0
    reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PortfolioContext:
    """Compact portfolio view handed to a strategy along with the market snapshot."""

    cash: float
    equity: float
    open_positions: int
    max_positions: int
    daily_pnl_pct: float
    position: dict | None = None  # for the symbol being evaluated
    extra: dict = field(default_factory=dict)
