import numpy as np
import pandas as pd
import pytest

from ai_trader.backtest import run_backtest
from ai_trader.indicators import add_indicators
from ai_trader.ml import (
    FEATURE_TAIL,
    MLModel,
    MLStrategy,
    make_dataset,
    make_features,
    model_path,
    roc_auc,
    time_series_splits,
    train,
)
from ai_trader.models import PortfolioContext
from tests.conftest import make_ohlcv

FAST = {"num_boost_round": 60}


def ar_ohlcv(n: int = 3000, phi: float = 0.3, seed: int = 1) -> pd.DataFrame:
    """Returns follow AR(1): the last return predicts the next one, so there is something to learn."""
    rng = np.random.default_rng(seed)
    eps = rng.normal(0, 0.01, n)
    rets = np.zeros(n)
    for i in range(1, n):
        rets[i] = phi * rets[i - 1] + eps[i]
    close = 100 * np.exp(np.cumsum(rets))
    open_ = np.concatenate([[100.0], close[:-1]])
    high = np.maximum(open_, close) * 1.002
    low = np.minimum(open_, close) * 0.998
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": rng.uniform(100, 1000, n)}, index=idx)


def ctx(position=None) -> PortfolioContext:
    return PortfolioContext(cash=10_000, equity=10_000, open_positions=0, max_positions=2, daily_pnl_pct=0.0, position=position)


def test_features_are_causal(ohlcv):
    full = make_features(add_indicators(ohlcv))
    part = make_features(add_indicators(ohlcv.iloc[:400]))
    pd.testing.assert_series_equal(full.iloc[399], part.iloc[-1])


def test_feature_tail_is_enough():
    df = add_indicators(make_ohlcv(n=500))
    pd.testing.assert_series_equal(make_features(df).iloc[-1], make_features(df.tail(FEATURE_TAIL)).iloc[-1])


def test_labels_match_forward_return(ohlcv):
    df = add_indicators(ohlcv)
    X, y, fwd = make_dataset(df, horizon=5, cost=0.003)
    assert X.index.equals(y.index) and y.index.equals(fwd.index)
    assert X.index[0] == df.index[200]  # warm-up dropped
    assert X.index[-1] == df.index[-6]  # last 5 bars have no full horizon
    t = X.index[100]
    i = df.index.get_loc(t)
    expected = df["close"].iloc[i + 5] / df["close"].iloc[i] - 1
    assert fwd[t] == pytest.approx(expected)
    assert y[t] == int(expected > 0.003)


def test_roc_auc():
    y = np.array([0, 0, 1, 1])
    assert roc_auc(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert roc_auc(y, np.array([0.9, 0.8, 0.2, 0.1])) == 0.0
    assert roc_auc(y, np.array([0.5, 0.5, 0.5, 0.5])) == 0.5
    assert roc_auc(y, np.array([0.1, 0.4, 0.35, 0.8])) == 0.75
    assert np.isnan(roc_auc(np.array([1, 1]), np.array([0.2, 0.3])))


def test_time_series_splits_never_train_on_the_future():
    splits = time_series_splits(100, n_splits=4, gap=3)
    assert len(splits) == 4
    tests = [te for _, te in splits]
    assert np.array_equal(np.concatenate(tests), np.arange(20, 100))  # contiguous, non-overlapping blocks
    for tr, te in splits:
        assert tr[0] == 0 and tr[-1] + 3 < te[0]
    with pytest.raises(ValueError):
        time_series_splits(10, n_splits=5, gap=3)


def test_training_labels_never_reach_the_holdout():
    raw = ar_ohlcv(n=1500)
    holdout_start = raw.index[1200]
    m, rep = train(raw, "TEST/USDT", "1h", horizon=6, cost=0.0, holdout_start=holdout_start, n_splits=3, params=FAST)
    assert m.oos_start == holdout_start.isoformat()
    last_train = raw.index.get_loc(pd.Timestamp(m.meta["train_end"]))
    assert last_train + 6 < 1200
    assert rep.holdout is not None and rep.holdout["start"] == str(holdout_start.date())


def test_learns_a_real_signal():
    _, rep = train(ar_ohlcv(), "TEST/USDT", "1h", horizon=1, cost=0.0, n_splits=4, params=FAST)
    aucs = [f["auc"] for f in rep.folds]
    assert min(aucs) > 0.55, aucs
    assert rep.importance[0][0] in {"ret_1", "dist_ema20", "macd_hist_pct", "bb_pos"}


def test_finds_nothing_in_noise():
    _, rep = train(ar_ohlcv(phi=0.0), "TEST/USDT", "1h", horizon=1, cost=0.0, n_splits=4, params=FAST)
    assert 0.45 < np.mean([f["auc"] for f in rep.folds]) < 0.55


def test_too_little_data_is_an_error():
    with pytest.raises(ValueError, match="training rows"):
        train(make_ohlcv(n=400), "TEST/USDT", "1h", params=FAST)


def test_save_load_roundtrip(tmp_path):
    m, _ = train(ar_ohlcv(n=1200), "BTC/USDT", "1h", horizon=1, cost=0.0, n_splits=2, params=FAST)
    path = model_path(tmp_path, "BTC/USDT", "1h")
    assert path.name == "BTC_USDT_1h.json"
    m.save(path)
    loaded = MLModel.load(path)
    X = make_features(add_indicators(ar_ohlcv(n=300, seed=5))).tail(10)
    np.testing.assert_allclose(loaded.predict_proba(X), m.predict_proba(X))


def test_strategy_without_model_holds(tmp_path):
    d = MLStrategy(tmp_path).decide("TEST/USDT", add_indicators(make_ohlcv()), ctx(), "1h")
    assert d.action == "hold" and "train" in d.reasoning


class FixedP:
    def __init__(self, p):
        self.p = p

    def predict(self, X):
        return np.full(len(X), self.p)


def strategy_with(p: float, tmp_path, horizon: int = 12) -> MLStrategy:
    s = MLStrategy(tmp_path, threshold=0.6)
    cols = list(make_features(add_indicators(make_ohlcv(n=300))).columns)
    s._models[("TEST/USDT", "1h")] = MLModel(FixedP(p), cols, "TEST/USDT", "1h", horizon, 0.003, None)
    return s


def test_strategy_thresholds(tmp_path):
    df = add_indicators(make_ohlcv())
    assert strategy_with(0.7, tmp_path).decide("TEST/USDT", df, ctx(), "1h").action == "buy"
    assert strategy_with(0.55, tmp_path).decide("TEST/USDT", df, ctx(), "1h").action == "hold"

    fresh = {"opened_at": df.index[-2].isoformat()}
    assert strategy_with(0.5, tmp_path).decide("TEST/USDT", df, ctx(fresh), "1h").action == "hold"
    sell = strategy_with(0.3, tmp_path).decide("TEST/USDT", df, ctx(fresh), "1h")
    assert sell.action == "sell" and sell.confidence == pytest.approx(0.7)


def test_strategy_exits_when_horizon_elapsed(tmp_path):
    df = add_indicators(make_ohlcv())
    old = {"opened_at": df.index[-13].isoformat()}
    d = strategy_with(0.5, tmp_path, horizon=12).decide("TEST/USDT", df, ctx(old), "1h")
    assert d.action == "sell" and "horizon" in d.reasoning
    # a fresh signal keeps the position even after the horizon
    assert strategy_with(0.7, tmp_path, horizon=12).decide("TEST/USDT", df, ctx(old), "1h").action == "hold"


def test_ml_backtest_runs_out_of_sample(settings, tmp_path):
    raw = ar_ohlcv(n=2000)
    m, _ = train(raw, "TEST/USDT", "1h", horizon=3, cost=0.0, holdout_start=raw.index[1500], n_splits=3, params=FAST)
    m.save(model_path(tmp_path, "TEST/USDT", "1h"))
    s = MLStrategy(tmp_path, threshold=settings.min_confidence)
    r = run_backtest(raw, s, settings, "TEST/USDT", "1h", warmup=1500)
    assert r.bars == 500
    assert sum(f.side == "buy" for f in r.fills) == sum(f.side == "sell" for f in r.fills)
    assert r.final_equity == pytest.approx(r.start_cash + sum(f.pnl for f in r.fills), abs=1e-6)
