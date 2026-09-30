"""Strategies share one interface: decide(symbol, df_with_indicators, ctx) -> Decision."""
from __future__ import annotations

from typing import Protocol

import pandas as pd

from .indicators import snapshot
from .models import Decision, PortfolioContext


class Strategy(Protocol):
    name: str

    def decide(self, symbol: str, df: pd.DataFrame, ctx: PortfolioContext, timeframe: str) -> Decision: ...


class RuleStrategy:
    """Cheap deterministic baseline: EMA trend + RSI pullback + MACD confirmation.

    Used for backtests without API cost and as a sanity benchmark for the Claude strategy.
    """

    name = "rule"

    def __init__(self, rsi_buy_max: float = 65.0, rsi_sell_min: float = 75.0):
        self.rsi_buy_max = rsi_buy_max
        self.rsi_sell_min = rsi_sell_min

    def decide(self, symbol: str, df: pd.DataFrame, ctx: PortfolioContext, timeframe: str) -> Decision:
        last, prev = df.iloc[-1], df.iloc[-2]
        uptrend = last["ema20"] > last["ema50"] > last["ema200"]
        macd_cross_up = prev["macd_hist"] <= 0 < last["macd_hist"]
        macd_cross_down = prev["macd_hist"] >= 0 > last["macd_hist"]
        atr_pct = float(last["atr_pct"]) if pd.notna(last["atr_pct"]) else 0.02

        if ctx.position is None:
            if uptrend and macd_cross_up and last["rsi14"] < self.rsi_buy_max:
                conf = 0.65 + 0.1 * (last["rsi14"] < 55)
                return Decision(
                    action="buy",
                    confidence=conf,
                    size_pct=min(1.0, 0.02 / max(atr_pct, 0.005)),
                    stop_loss_pct=max(0.02, 2.0 * atr_pct),
                    take_profit_pct=max(0.04, 4.0 * atr_pct),
                    reasoning=f"EMA20>EMA50>EMA200 uptrend, MACD histogram crossed up, RSI {last['rsi14']:.0f}.",
                    key_risks=["Trend reversal", "False MACD cross"],
                )
            return Decision.hold("No entry setup")

        if macd_cross_down or last["rsi14"] > self.rsi_sell_min or last["close"] < last["ema50"]:
            return Decision(
                action="sell",
                confidence=0.7,
                size_pct=1.0,
                reasoning="Exit: MACD crossed down, RSI overbought, or close below EMA50.",
            )
        return Decision.hold("Holding open position")


class ClaudeStrategy:
    name = "claude"

    def __init__(self, brain):
        self.brain = brain

    def decide(self, symbol: str, df: pd.DataFrame, ctx: PortfolioContext, timeframe: str) -> Decision:
        return self.brain.decide(snapshot(df, symbol, timeframe), ctx)
