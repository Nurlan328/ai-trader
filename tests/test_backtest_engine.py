import pandas as pd
import pytest

from ai_trader.backtest import run_backtest
from ai_trader.broker import PaperBroker, Portfolio
from ai_trader.engine import TradingEngine
from ai_trader.journal import Journal
from ai_trader.models import Decision
from ai_trader.strategy import RuleStrategy
from tests.conftest import make_ohlcv


def test_rule_backtest_runs_and_accounts(ohlcv, settings):
    r = run_backtest(ohlcv, RuleStrategy(), settings, "TEST/USDT", "1h", warmup=200)
    assert r.bars == len(ohlcv) - 200
    assert r.final_equity == pytest.approx(r.start_cash * (1 + r.total_return))
    assert -1 <= r.max_drawdown <= 0
    assert len(r.equity_curve) == r.bars
    # every buy has a matching sell (positions are closed at the end)
    buys = sum(f.side == "buy" for f in r.fills)
    sells = sum(f.side == "sell" for f in r.fills)
    assert buys == sells
    # cash accounting: final equity == start + realized pnl (no open positions left)
    assert r.final_equity == pytest.approx(r.start_cash + sum(f.pnl for f in r.fills), abs=1e-6)


def test_backtest_every_n_reduces_strategy_calls(ohlcv, settings):
    calls = {"n": 0}

    class Counting(RuleStrategy):
        def decide(self, *a, **k):
            calls["n"] += 1
            return super().decide(*a, **k)

    run_backtest(ohlcv, Counting(), settings, "TEST/USDT", "1h", warmup=200, every=10)
    assert calls["n"] == (len(ohlcv) - 200 + 9) // 10


class AlwaysBuy:
    name = "always_buy"

    def decide(self, symbol, df, ctx, timeframe):
        return Decision(action="buy", confidence=0.9, size_pct=1.0, stop_loss_pct=0.05, take_profit_pct=0.10, reasoning="t")


def test_stop_loss_and_take_profit_exit_in_backtest(settings):
    # strongly trending up series -> take profits should trigger
    df = make_ohlcv(n=400, drift=0.003, vol=0.005)
    r = run_backtest(df, AlwaysBuy(), settings, "TEST/USDT", "1h", warmup=100)
    reasons = {f.reason for f in r.fills if f.side == "sell"}
    assert "take_profit" in reasons
    assert r.total_return > 0


def test_engine_tick_paper(settings, ohlcv, tmp_path):
    portfolio = Portfolio(cash=settings.start_cash, start_cash=settings.start_cash)
    broker = PaperBroker(portfolio, settings.fee_pct, settings.slippage_pct)
    journal = Journal(":memory:")
    engine = TradingEngine(settings, AlwaysBuy(), broker, portfolio, journal, fetch=lambda sym: ohlcv, state_path=tmp_path / "p.json")

    fills = engine.tick()
    assert len(fills) == 1 and fills[0].side == "buy"
    assert "TEST/USDT" in portfolio.positions
    assert (tmp_path / "p.json").exists()
    assert journal.recent_decisions(5)[0]["approved"] == 1

    # second tick: already in position -> buy rejected, nothing happens
    assert engine.tick() == []
    assert journal.recent_decisions(1)[0]["risk_reason"].startswith("already in position")


def test_engine_skips_symbol_on_fetch_error(settings):
    def boom(sym):
        raise ConnectionError("down")

    portfolio = Portfolio(cash=1000, start_cash=1000)
    engine = TradingEngine(settings, RuleStrategy(), PaperBroker(portfolio), portfolio, Journal(), fetch=boom)
    assert engine.tick() == []
