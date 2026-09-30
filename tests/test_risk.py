import pytest

from ai_trader.broker import PaperBroker, Portfolio
from ai_trader.models import Decision
from ai_trader.risk import RiskManager


def buy(conf=0.8, size=1.0, sl=None, tp=None):
    return Decision(action="buy", confidence=conf, size_pct=size, stop_loss_pct=sl, take_profit_pct=tp, reasoning="t")


def test_hold_is_never_approved(settings):
    v = RiskManager(settings).evaluate(Decision.hold("x"), "A/USDT", 100, Portfolio(10_000, 10_000), 10_000)
    assert not v.approved


def test_low_confidence_rejected(settings):
    v = RiskManager(settings).evaluate(buy(conf=0.5), "A/USDT", 100, Portfolio(10_000, 10_000), 10_000)
    assert not v.approved and "confidence" in v.reason


def test_buy_size_capped_by_max_position_pct(settings):
    v = RiskManager(settings).evaluate(buy(size=1.0, sl=0.05, tp=0.1), "A/USDT", 100, Portfolio(10_000, 10_000), 10_000)
    assert v.approved
    assert v.notional == pytest.approx(10_000 * 0.2)
    assert v.stop_loss == pytest.approx(95)
    assert v.take_profit == pytest.approx(110)


def test_buy_size_scaled_by_size_pct_and_defaults(settings):
    v = RiskManager(settings).evaluate(buy(size=0.5), "A/USDT", 100, Portfolio(10_000, 10_000), 10_000, atr_pct=0.04)
    assert v.notional == pytest.approx(1_000)
    # no stop given -> max(default 3%, 1.5*ATR 6%) = 6%
    assert v.stop_loss == pytest.approx(94)
    assert v.take_profit == pytest.approx(112)


def test_buy_limited_by_cash(settings):
    p = Portfolio(cash=50, start_cash=10_000)
    v = RiskManager(settings).evaluate(buy(), "A/USDT", 100, p, 10_000)
    assert v.approved
    assert v.notional < 50


def test_clamps_out_of_range_llm_values(settings):
    v = RiskManager(settings).evaluate(buy(conf=3.0, size=7.0, sl=0.9), "A/USDT", 100, Portfolio(10_000, 10_000), 10_000)
    assert v.approved
    assert v.decision.confidence == 1.0
    assert v.notional == pytest.approx(2_000)
    assert v.stop_loss == pytest.approx(50)  # sl clamped to 50%


def test_no_pyramiding_and_max_positions(settings):
    p = Portfolio(cash=10_000, start_cash=10_000)
    b = PaperBroker(p)
    b.buy("A/USDT", 1000, 100, "t")
    rm = RiskManager(settings)
    assert not rm.evaluate(buy(), "A/USDT", 100, p, 10_000).approved
    b.buy("B/USDT", 1000, 100, "t")
    v = rm.evaluate(buy(), "C/USDT", 100, p, 10_000)
    assert not v.approved and "max open" in v.reason


def test_daily_loss_limit_blocks_new_buys(settings):
    p = Portfolio(cash=9_600, start_cash=10_000)
    p.roll_day("2025-01-01T00:00:00", 10_000)
    v = RiskManager(settings).evaluate(buy(), "A/USDT", 100, p, 9_600)
    assert not v.approved and "daily loss" in v.reason


def test_sell_requires_position_and_avoids_dust(settings):
    p = Portfolio(cash=10_000, start_cash=10_000)
    rm = RiskManager(settings)
    sell = Decision(action="sell", confidence=0.9, size_pct=0.95, reasoning="t")
    assert not rm.evaluate(sell, "A/USDT", 100, p, 10_000).approved
    PaperBroker(p, fee_pct=0, slippage_pct=0).buy("A/USDT", 100, 100, "t")  # qty 1.0
    v = rm.evaluate(sell, "A/USDT", 100, p, 10_000)
    assert v.approved
    assert v.qty == pytest.approx(1.0)  # remaining 5 units of value < min_notional -> close all
