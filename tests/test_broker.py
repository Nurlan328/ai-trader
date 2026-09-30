import pytest

from ai_trader.broker import PaperBroker, Portfolio


def test_buy_then_sell_pnl_with_fees():
    p = Portfolio(cash=1000, start_cash=1000)
    b = PaperBroker(p, fee_pct=0.001, slippage_pct=0.0)
    f = b.buy("X/USDT", notional=500, price=100, ts="2025-01-01T00:00:00+00:00")
    assert f.qty == pytest.approx(5.0)
    assert p.cash == pytest.approx(1000 - 500 - 0.5)
    assert "X/USDT" in p.positions

    f2 = b.sell("X/USDT", qty=5.0, price=110, ts="2025-01-01T01:00:00+00:00")
    assert f2.pnl == pytest.approx(50 - 0.55 - 0.5)  # sell fee and entry fee
    assert p.cash == pytest.approx(1000 - 500 - 0.5 + 550 - 0.55)
    assert "X/USDT" not in p.positions
    assert p.realized_pnl == pytest.approx(f2.pnl)
    assert p.n_trades == 2


def test_partial_sell_keeps_remainder():
    p = Portfolio(cash=1000, start_cash=1000)
    b = PaperBroker(p, fee_pct=0.0, slippage_pct=0.0)
    b.buy("X/USDT", 400, 100, "t")
    b.sell("X/USDT", 1.0, 120, "t")
    assert p.positions["X/USDT"].qty == pytest.approx(3.0)


def test_insufficient_cash_raises():
    p = Portfolio(cash=100, start_cash=100)
    b = PaperBroker(p, fee_pct=0.001)
    with pytest.raises(ValueError):
        b.buy("X/USDT", 100, 10, "t")  # fee pushes total over cash


def test_slippage_applied_against_trader():
    p = Portfolio(cash=1000, start_cash=1000)
    b = PaperBroker(p, fee_pct=0.0, slippage_pct=0.01)
    f = b.buy("X/USDT", 100, 100, "t")
    assert f.price == pytest.approx(101)
    f2 = b.sell("X/USDT", f.qty, 100, "t")
    assert f2.price == pytest.approx(99)


def test_portfolio_roundtrip_persistence(tmp_path):
    p = Portfolio(cash=1000, start_cash=1000)
    PaperBroker(p).buy("X/USDT", 100, 50, "t", stop_loss=45, take_profit=60)
    path = tmp_path / "p.json"
    p.save(path)
    q = Portfolio.load_or_create(path, 999)
    assert q.cash == pytest.approx(p.cash)
    assert q.positions["X/USDT"].stop_loss == 45
    assert q.positions["X/USDT"].take_profit == 60


def test_daily_pnl_rolls_on_new_day():
    p = Portfolio(cash=1000, start_cash=1000)
    p.roll_day("2025-01-01T10:00:00", 1000)
    assert p.daily_pnl_pct(950) == pytest.approx(-0.05)
    p.roll_day("2025-01-02T00:00:00", 950)
    assert p.daily_pnl_pct(950) == pytest.approx(0.0)
