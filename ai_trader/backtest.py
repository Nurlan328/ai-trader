"""Bar-by-bar backtest that reuses the same risk manager and paper broker as live trading."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .broker import PaperBroker, Portfolio
from .config import Settings
from .data import bars_per_year
from .engine import build_context, check_exit
from .indicators import add_indicators
from .models import Fill
from .risk import RiskManager
from .strategy import Strategy

log = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    symbol: str
    strategy: str
    bars: int
    start: str
    end: str
    start_cash: float
    final_equity: float
    total_return: float
    buy_hold_return: float
    max_drawdown: float
    sharpe: float
    n_trades: int
    n_round_trips: int
    win_rate: float
    profit_factor: float
    fees_paid: float
    fills: list[Fill] = field(default_factory=list)
    equity_curve: pd.Series = field(default_factory=pd.Series)

    def summary(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if k not in ("fills", "equity_curve")}


def run_backtest(
    df: pd.DataFrame,
    strategy: Strategy,
    settings: Settings,
    symbol: str,
    timeframe: str,
    warmup: int = 200,
    every: int = 1,
) -> BacktestResult:
    """`every` = call the strategy only every N bars (stops/targets are still checked every bar).
    Useful to bound API cost when the strategy is Claude."""
    df = add_indicators(df)
    if len(df) <= warmup + 2:
        raise ValueError(f"need more than {warmup + 2} bars, got {len(df)}")

    portfolio = Portfolio(cash=settings.start_cash, start_cash=settings.start_cash)
    broker = PaperBroker(portfolio, settings.fee_pct, settings.slippage_pct)
    risk = RiskManager(settings)
    fills: list[Fill] = []
    equity_points: list[float] = []

    for i in range(warmup, len(df)):
        window = df.iloc[: i + 1]
        bar = window.iloc[-1]
        price = float(bar["close"])
        ts = window.index[-1].isoformat()
        prices = {symbol: price}
        equity = portfolio.equity(prices)
        portfolio.roll_day(ts, equity)

        exit_fill = check_exit(portfolio, broker, symbol, bar, ts)
        if exit_fill:
            fills.append(exit_fill)
            equity = portfolio.equity(prices)

        if (i - warmup) % every == 0:
            ctx = build_context(portfolio, symbol, price, equity, settings.max_open_positions)
            decision = strategy.decide(symbol, window, ctx, timeframe)
            atr_pct = float(bar["atr_pct"]) if pd.notna(bar["atr_pct"]) else None
            verdict = risk.evaluate(decision, symbol, price, portfolio, equity, atr_pct)
            if verdict.approved:
                if decision.action == "buy":
                    fills.append(broker.buy(symbol, verdict.notional, price, ts, verdict.stop_loss, verdict.take_profit, reason=strategy.name))
                else:
                    fills.append(broker.sell(symbol, verdict.qty, price, ts, reason=strategy.name))

        equity_points.append(portfolio.equity(prices))

    # Close any open position at the last close so results are comparable.
    last_price = float(df["close"].iloc[-1])
    if symbol in portfolio.positions:
        fills.append(broker.sell(symbol, portfolio.positions[symbol].qty, last_price, df.index[-1].isoformat(), reason="end_of_test"))
        equity_points[-1] = portfolio.equity({symbol: last_price})

    curve = pd.Series(equity_points, index=df.index[warmup:])
    returns = curve.pct_change().dropna()
    sharpe = float(returns.mean() / returns.std() * np.sqrt(bars_per_year(timeframe))) if len(returns) > 1 and returns.std() > 0 else 0.0
    drawdown = (curve / curve.cummax() - 1).min()
    sells = [f for f in fills if f.side == "sell"]
    wins = [f.pnl for f in sells if f.pnl > 0]
    losses = [-f.pnl for f in sells if f.pnl <= 0]
    first_close = float(df["close"].iloc[warmup])

    return BacktestResult(
        symbol=symbol,
        strategy=strategy.name,
        bars=len(df) - warmup,
        start=str(df.index[warmup].date()),
        end=str(df.index[-1].date()),
        start_cash=settings.start_cash,
        final_equity=float(curve.iloc[-1]),
        total_return=float(curve.iloc[-1] / settings.start_cash - 1),
        buy_hold_return=float(last_price / first_close - 1),
        max_drawdown=float(drawdown),
        sharpe=sharpe,
        n_trades=len(fills),
        n_round_trips=len(sells),
        win_rate=len(wins) / len(sells) if sells else 0.0,
        profit_factor=(sum(wins) / sum(losses)) if losses and sum(losses) > 0 else (float("inf") if wins else 0.0),
        fees_paid=portfolio.fees_paid,
        fills=fills,
        equity_curve=curve,
    )
