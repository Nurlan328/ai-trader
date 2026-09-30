"""LightGBM strategy: learn P(return over the next `horizon` bars beats round-trip costs) from indicator features.

Train offline with `python -m ai_trader train SYMBOL`: it runs a walk-forward check, keeps the last
`--holdout-days` out of training and saves the model. MLStrategy then loads one model per symbol/timeframe.
Uses LightGBM's native API rather than scikit-learn, so the only compiled dependency is LightGBM itself.

Everything is causal: features at bar t use bars <= t, the label at t uses bars t+1..t+horizon. That is why
walk-forward folds keep a `horizon`-bar gap and training stops `horizon` bars before the holdout starts.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .indicators import add_indicators
from .models import Decision, PortfolioContext

WARMUP_BARS = 200  # ema200 is meaningless before this
FEATURE_TAIL = 40  # bars make_features needs on top of add_indicators() for the last row to be exact
MIN_TRAIN_ROWS = 300

DEFAULT_PARAMS: dict = dict(
    objective="binary",
    num_boost_round=300,
    learning_rate=0.03,
    num_leaves=15,
    max_depth=4,
    min_data_in_leaf=50,  # market data is noise-heavy: forbid tiny leaves
    bagging_fraction=0.8,
    bagging_freq=1,
    feature_fraction=0.8,
    lambda_l2=1.0,
    seed=42,
    deterministic=True,
    verbose=-1,
)


# --- dataset ----------------------------------------------------------------

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Scale-free features from an add_indicators() frame, so one recipe fits BTC at 60k and a coin at 0.3."""
    c = df["close"]
    width = (df["hh_20"] - df["ll_20"]).replace(0, np.nan)
    vol_ratio = df["vol_ratio"].where(df["vol_ratio"] > 0)
    f = pd.DataFrame(
        {
            "ret_1": df["ret_1"],
            "ret_5": df["ret_5"],
            "ret_20": df["ret_20"],
            "vol_20": df["ret_1"].rolling(20).std(),
            "rsi14": df["rsi14"],
            "macd_hist_pct": df["macd_hist"] / c,
            "atr_pct": df["atr_pct"],
            "bb_pos": df["bb_pos"],
            "log_vol_ratio": np.log(vol_ratio),  # volume spikes are multiplicative
            "dist_ema20": c / df["ema20"] - 1,
            "ema20_vs_50": df["ema20"] / df["ema50"] - 1,
            "ema50_vs_200": df["ema50"] / df["ema200"] - 1,
            "range_pos_20": (c - df["ll_20"]) / width,
        },
        index=df.index,
    )
    return f.replace([np.inf, -np.inf], np.nan)


def forward_return(close: pd.Series, horizon: int) -> pd.Series:
    return close.shift(-horizon) / close - 1


def make_dataset(df: pd.DataFrame, horizon: int, cost: float) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """X, y, forward return for every bar after warm-up that has a full `horizon` of future bars.

    y = 1 when the move over the next `horizon` bars pays for the round trip (fees + slippage)."""
    fwd = forward_return(df["close"], horizon)
    X = make_features(df)
    keep = fwd.notna()
    keep.iloc[: min(WARMUP_BARS, len(keep))] = False
    fwd = fwd[keep]
    return X[keep], (fwd > cost).astype(int), fwd


# --- evaluation ---------------------------------------------------------------

def _fit(X: pd.DataFrame, y: pd.Series, params: dict | None = None):
    import lightgbm as lgb

    p = {**DEFAULT_PARAMS, **(params or {})}
    rounds = p.pop("num_boost_round")
    return lgb.train(p, lgb.Dataset(X, label=y), num_boost_round=rounds)


def roc_auc(y: np.ndarray, p: np.ndarray) -> float:
    """Probability that a random positive scores above a random negative (Mann-Whitney U), ties count half."""
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(p).rank().to_numpy()
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def time_series_splits(n: int, n_splits: int, gap: int = 0) -> list[tuple[np.ndarray, np.ndarray]]:
    """Expanding-window splits: n_splits equal test blocks at the end, each trained on everything `gap` bars before it."""
    test_size = n // (n_splits + 1)
    if test_size < 1 or test_size - gap < 1:
        raise ValueError(f"{n} rows are too few for {n_splits} splits with gap {gap}")
    splits = []
    for k in range(n_splits):
        start = n - (n_splits - k) * test_size
        splits.append((np.arange(0, start - gap), np.arange(start, start + test_size)))
    return splits


def evaluate(p: np.ndarray, y: pd.Series, fwd: pd.Series, threshold: float) -> dict:
    """Classifier quality plus what actually matters for trading: the forward return of the bars we would buy."""
    yt = y.to_numpy()
    ft = fwd.to_numpy()
    picked = p >= threshold
    return {
        "start": str(y.index[0].date()),
        "end": str(y.index[-1].date()),
        "n": len(yt),
        "auc": roc_auc(yt, p),
        "accuracy": float(((p >= 0.5) == yt).mean()),
        "base_rate": float(yt.mean()),
        "n_signals": int(picked.sum()),
        "hit_rate": float(yt[picked].mean()) if picked.any() else float("nan"),
        "fwd_ret_signals": float(ft[picked].mean()) if picked.any() else float("nan"),
        "fwd_ret_all": float(ft.mean()),
    }


def walk_forward(
    X: pd.DataFrame, y: pd.Series, fwd: pd.Series, horizon: int, threshold: float, n_splits: int = 5, params: dict | None = None
) -> list[dict]:
    """Train on the past, test on the next block, roll forward. The gap drops train rows whose label peeks into the test block."""
    folds = []
    for k, (tr, te) in enumerate(time_series_splits(len(X), n_splits, gap=horizon), 1):
        model = _fit(X.iloc[tr], y.iloc[tr], params)
        p = model.predict(X.iloc[te])
        folds.append({"fold": k, "n_train": len(tr), **evaluate(p, y.iloc[te], fwd.iloc[te], threshold)})
    return folds


# --- model ------------------------------------------------------------------

@dataclass
class MLModel:
    model: object  # lightgbm.Booster (anything with predict(X) -> P(y=1))
    features: list[str]
    symbol: str
    timeframe: str
    horizon: int
    cost: float
    oos_start: str | None  # first bar no training row ever saw (features or label); None = trained on everything
    meta: dict = field(default_factory=dict)

    def predict_proba(self, feats: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.model.predict(feats[self.features]), dtype=float)

    def save(self, path: Path) -> None:
        """Plain JSON with LightGBM's text model inside: readable, and loading it runs no code (unlike pickle)."""
        doc = {k: v for k, v in self.__dict__.items() if k != "model"}
        doc["booster"] = self.model.model_to_string()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(doc), encoding="utf-8")

    @staticmethod
    def load(path: Path) -> "MLModel":
        import lightgbm as lgb

        doc = json.loads(Path(path).read_text(encoding="utf-8"))
        return MLModel(model=lgb.Booster(model_str=doc.pop("booster")), **doc)


def model_path(model_dir: Path, symbol: str, timeframe: str) -> Path:
    return Path(model_dir) / f"{re.sub(r'[^A-Za-z0-9]+', '_', symbol)}_{timeframe}.json"


@dataclass
class TrainReport:
    folds: list[dict]
    holdout: dict | None
    importance: list[tuple[str, float]]


def train(
    raw: pd.DataFrame,
    symbol: str,
    timeframe: str,
    horizon: int = 12,
    cost: float = 0.003,
    holdout_start: pd.Timestamp | None = None,
    threshold: float = 0.6,
    n_splits: int = 5,
    params: dict | None = None,
) -> tuple[MLModel, TrainReport]:
    """Walk-forward check on the training part, then fit the final model on all of it and score the holdout."""
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    df = add_indicators(raw)
    X, y, fwd = make_dataset(df, horizon, cost)
    pos = df.index.get_indexer(X.index)  # bar number of each row in df

    if holdout_start is not None:
        oos_pos = int(df.index.searchsorted(holdout_start))
        if oos_pos >= len(df):
            raise ValueError("holdout starts after the last bar")
        in_train = pos + horizon < oos_pos  # the label of a training row must end before the holdout
        in_holdout = pos >= oos_pos
        oos_start = df.index[oos_pos].isoformat()
    else:
        in_train = np.ones(len(X), dtype=bool)
        in_holdout = np.zeros(len(X), dtype=bool)
        oos_start = None

    Xtr, ytr, ftr = X[in_train], y[in_train], fwd[in_train]
    if len(Xtr) < MIN_TRAIN_ROWS:
        raise ValueError(f"need at least {MIN_TRAIN_ROWS} training rows after warm-up, got {len(Xtr)}; fetch more --days")
    if ytr.nunique() < 2:
        raise ValueError("training labels are all one class; nothing to learn")

    folds = walk_forward(Xtr, ytr, ftr, horizon, threshold, n_splits, params)
    model = _fit(Xtr, ytr, params)

    holdout = None
    if in_holdout.any():
        p = model.predict(X[in_holdout])
        holdout = evaluate(p, y[in_holdout], fwd[in_holdout], threshold)

    gains = np.asarray(model.feature_importance(importance_type="gain"), dtype=float)
    share = gains / gains.sum() if gains.sum() > 0 else gains
    importance = sorted(zip(X.columns, share.tolist()), key=lambda kv: -kv[1])

    import lightgbm

    m = MLModel(
        model=model,
        features=list(X.columns),
        symbol=symbol,
        timeframe=timeframe,
        horizon=horizon,
        cost=cost,
        oos_start=oos_start,
        meta={
            "train_start": Xtr.index[0].isoformat(),
            "train_end": Xtr.index[-1].isoformat(),
            "n_train": len(Xtr),
            "base_rate": float(ytr.mean()),
            "cv_auc_mean": float(np.nanmean([f["auc"] for f in folds])),
            "params": {**DEFAULT_PARAMS, **(params or {})},
            "lightgbm": lightgbm.__version__,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    return m, TrainReport(folds, holdout, importance)


# --- strategy ---------------------------------------------------------------

def _bars_held(df: pd.DataFrame, opened_at: str | None) -> int | None:
    if not opened_at:
        return None
    try:
        ts = pd.Timestamp(opened_at)
    except ValueError:
        return None
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return int((df.index > ts).sum())


class MLStrategy:
    """Buy when P(up beyond costs) >= threshold; exit when it drops to 1 - threshold or the horizon runs out.

    confidence is the model's probability as is, so the risk manager's MIN_CONFIDENCE gates it too."""

    name = "ml"

    def __init__(self, model_dir: Path, threshold: float = 0.6):
        self.model_dir = Path(model_dir)
        self.threshold = threshold
        self._models: dict[tuple[str, str], MLModel | None] = {}

    def model_for(self, symbol: str, timeframe: str) -> MLModel | None:
        key = (symbol, timeframe)
        if key not in self._models:
            path = model_path(self.model_dir, symbol, timeframe)
            self._models[key] = MLModel.load(path) if path.exists() else None
        return self._models[key]

    def decide(self, symbol: str, df: pd.DataFrame, ctx: PortfolioContext, timeframe: str) -> Decision:
        m = self.model_for(symbol, timeframe)
        if m is None:
            return Decision.hold(f"No ML model for {symbol} {timeframe}; run: python -m ai_trader train {symbol} --timeframe {timeframe}")
        p = float(m.predict_proba(make_features(df.tail(FEATURE_TAIL)).iloc[[-1]])[0])
        t = self.threshold
        what = f"P(+{m.horizon} bars beats {m.cost:.2%} costs)={p:.2f}"

        if ctx.position is None:
            if p < t:
                return Decision.hold(f"{what} < {t:.2f}")
            atr_pct = float(df["atr_pct"].iloc[-1]) if pd.notna(df["atr_pct"].iloc[-1]) else 0.02
            return Decision(
                action="buy",
                confidence=p,
                size_pct=min(1.0, 0.02 / max(atr_pct, 0.005)),
                reasoning=f"LightGBM {what} >= {t:.2f}.",
                key_risks=["The edge is statistical: any single trade is close to a coin flip", "Market regime changed since training"],
            )

        if p <= 1 - t:
            return Decision(action="sell", confidence=1 - p, size_pct=1.0, reasoning=f"LightGBM {what} <= {1 - t:.2f}: expects a move against the position.")
        held = _bars_held(df, ctx.position.get("opened_at"))
        if held is not None and held >= m.horizon and p < t:
            return Decision(action="sell", confidence=1.0, size_pct=1.0, reasoning=f"Held {held} bars >= horizon {m.horizon} with no fresh signal ({what}).")
        return Decision.hold(f"Holding; {what}")
