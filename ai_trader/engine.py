"""One tick = for each symbol: fetch -> indicators -> exits -> strategy -> risk -> execute -> journal."""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable

import pandas as pd

from .broker import PaperBroker, Portfolio
from .config import Settings
from .indicators import add_indicators
from .journal import Journal
from .models import Fill, PortfolioContext
from .risk import RiskManager
from .strategy import Strategy

log = logging.getLogger(__name__)

FetchFn = Callable[[str], pd.DataFrame]


def build_context(portfolio: Portfolio, symbol: str, price: float, equity: float, max_positions: int) -> PortfolioContext:
    pos = portfolio.positions.get(symbol)
    return PortfolioContext(
        cash=round(portfolio.cash, 2),
        equity=round(equity, 2),
        open_positions=len(portfolio.positions),
        max_positions=max_positions,
        daily_pnl_pct=round(portfolio.daily_pnl_pct(equity) * 100, 2),
        position=None
        if pos is None
        else {
            "qty": pos.qty,
            "entry_price": pos.entry_price,
            "opened_at": pos.opened_at,
            "unrealized_pnl_pct": round(pos.unrealized_pct(price) * 100, 2),
            "stop_loss": pos.stop_loss,
            "take_profit": pos.take_profit,
        },
    )


def check_exit(portfolio: Portfolio, broker: PaperBroker, symbol: str, bar: pd.Series, ts: str) -> Fill | None:
    """Close a position if this bar crossed its stop or target. Stop is checked first (conservative)."""
    pos = portfolio.positions.get(symbol)
    if pos is None:
        return None
    if pos.stop_loss is not None and float(bar["low"]) <= pos.stop_loss:
        return broker.sell(symbol, pos.qty, pos.stop_loss, ts, reason="stop_loss")
    if pos.take_profit is not None and float(bar["high"]) >= pos.take_profit:
        return broker.sell(symbol, pos.qty, pos.take_profit, ts, reason="take_profit")
    return None


class TradingEngine:
    def __init__(
        self,
        settings: Settings,
        strategy: Strategy,
        broker: PaperBroker,
        portfolio: Portfolio,
        journal: Journal,
        fetch: FetchFn,
        state_path: Path | None = None,
    ):
        self.s = settings
        self.strategy = strategy
        self.broker = broker
        self.portfolio = portfolio
        self.journal = journal
        self.fetch = fetch
        self.risk = RiskManager(settings)
        self.state_path = state_path
        self.last_prices: dict[str, float] = {}

    def _save(self) -> None:
        if self.state_path:
            self.portfolio.save(self.state_path)

    def tick(self) -> list[Fill]:
        fills: list[Fill] = []
        frames: dict[str, pd.DataFrame] = {}
        for symbol in self.s.symbol_list:
            try:
                df = add_indicators(self.fetch(symbol))
            except Exception as e:  # network / exchange hiccup: skip this symbol this tick
                log.error("fetch failed for %s: %s", symbol, e)
                continue
            if len(df) < 60:
                log.warning("not enough bars for %s (%d), skipping", symbol, len(df))
                continue
            frames[symbol] = df
            self.last_prices[symbol] = float(df["close"].iloc[-1])

        for symbol, df in frames.items():
            bar = df.iloc[-1]
            price = float(bar["close"])
            ts = df.index[-1].isoformat()
            equity = self.portfolio.equity(self.last_prices)
            self.portfolio.roll_day(ts, equity)

            exit_fill = check_exit(self.portfolio, self.broker, symbol, bar, ts)
            if exit_fill:
                log.info("EXIT %s %s qty=%.6f @ %.4f pnl=%.2f", exit_fill.reason, symbol, exit_fill.qty, exit_fill.price, exit_fill.pnl)
                self.journal.log_fill(exit_fill)
                fills.append(exit_fill)
                equity = self.portfolio.equity(self.last_prices)

            ctx = build_context(self.portfolio, symbol, price, equity, self.s.max_open_positions)
            decision = self.strategy.decide(symbol, df, ctx, self.s.timeframe)
            atr_pct = float(bar["atr_pct"]) if pd.notna(bar["atr_pct"]) else None
            verdict = self.risk.evaluate(decision, symbol, price, self.portfolio, equity, atr_pct)
            self.journal.log_decision(ts, symbol, price, self.strategy.name, verdict.decision, verdict.approved, verdict.reason)
            log.info("%s %s conf=%.2f -> %s (%s)", symbol, decision.action.upper(), decision.confidence, "EXEC" if verdict.approved else "skip", verdict.reason)

            if not verdict.approved:
                continue
            try:
                if decision.action == "buy":
                    fill = self.broker.buy(symbol, verdict.notional, price, ts, verdict.stop_loss, verdict.take_profit, reason=self.strategy.name)
                else:
                    fill = self.broker.sell(symbol, verdict.qty, price, ts, reason=self.strategy.name)
            except Exception as e:
                log.error("order failed for %s: %s", symbol, e)
                continue
            self.journal.log_fill(fill)
            fills.append(fill)
            self._save()

        self._save()
        return fills

    def run_forever(self, interval_sec: int | None = None) -> None:
        interval = interval_sec or self.s.loop_interval_sec
        log.info("engine started: mode=%s strategy=%s symbols=%s every %ss", self.broker.name, self.strategy.name, self.s.symbol_list, interval)
        while True:
            started = time.time()
            try:
                self.tick()
            except KeyboardInterrupt:
                raise
            except Exception:
                log.exception("tick failed")
            equity = self.portfolio.equity(self.last_prices)
            log.info("equity=%.2f cash=%.2f positions=%d realized=%.2f", equity, self.portfolio.cash, len(self.portfolio.positions), self.portfolio.realized_pnl)
            sleep_for = max(5.0, interval - (time.time() - started))
            try:
                time.sleep(sleep_for)
            except KeyboardInterrupt:
                log.info("stopped by user")
                return
