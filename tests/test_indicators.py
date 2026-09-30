import numpy as np
import pandas as pd

from ai_trader.indicators import add_indicators, rsi, sma, snapshot


def test_sma_matches_manual():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    assert sma(s, 3).tolist()[2:] == [2.0, 3.0, 4.0]


def test_rsi_bounds_and_direction():
    up = pd.Series(np.linspace(100, 200, 60))
    down = pd.Series(np.linspace(200, 100, 60))
    r_up, r_down = rsi(up), rsi(down)
    assert ((r_up >= 0) & (r_up <= 100)).all()
    assert r_up.iloc[-1] > 90
    assert r_down.iloc[-1] < 10


def test_add_indicators_is_causal(ohlcv):
    """Indicators at bar i must not change when future bars are appended."""
    full = add_indicators(ohlcv)
    partial = add_indicators(ohlcv.iloc[:400])
    cols = ["ema20", "ema50", "rsi14", "macd_hist", "atr14", "bb_pos", "vol_ratio", "ret_5"]
    pd.testing.assert_frame_equal(full[cols].iloc[:400], partial[cols], check_exact=False, rtol=1e-9)


def test_snapshot_shape(ohlcv):
    df = add_indicators(ohlcv)
    snap = snapshot(df, "TEST/USDT", "1h", recent_bars=10)
    assert snap["symbol"] == "TEST/USDT"
    assert snap["price"] == float(df["close"].iloc[-1])
    assert len(snap["recent_candles"]) == 10
    assert snap["trend_ema_20_50_200"] in {"up", "down", "mixed"}
    assert 0 <= snap["rsi14"] <= 100
