import numpy as np
import pandas as pd
import pytest

from ai_trader.config import Settings


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        _env_file=None,
        start_cash=10_000,
        max_position_pct=0.2,
        max_open_positions=2,
        max_daily_loss_pct=0.03,
        min_confidence=0.6,
        min_notional=10,
        fee_pct=0.001,
        slippage_pct=0.0,
        state_dir=tmp_path,
        symbols="TEST/USDT",
        timeframe="1h",
    )


def make_ohlcv(n: int = 600, seed: int = 7, drift: float = 0.0005, vol: float = 0.01, start: float = 100.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(drift, vol, n)
    close = start * np.exp(np.cumsum(rets))
    open_ = np.concatenate([[start], close[:-1]])
    high = np.maximum(open_, close) * (1 + rng.uniform(0, vol, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, vol, n))
    volume = rng.uniform(100, 1000, n)
    idx = pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx)


@pytest.fixture
def ohlcv() -> pd.DataFrame:
    return make_ohlcv()
