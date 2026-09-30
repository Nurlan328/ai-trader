"""Technical indicators in plain pandas (no TA-Lib) and a compact snapshot for the LLM."""
from __future__ import annotations

import numpy as np
import pandas as pd


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    """Wilder RSI. Returns 50 while there is not enough data."""
    delta = s.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    out = out.where(loss != 0, 100.0)  # no losses at all -> max RSI
    out.iloc[: min(n, len(out))] = np.nan
    return out.fillna(50.0)


def macd(s: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[pd.Series, pd.Series, pd.Series]:
    line = ema(s, fast) - ema(s, slow)
    sig = ema(line, signal)
    return line, sig, line - sig


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def bollinger(s: pd.Series, n: int = 20, k: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series]:
    mid = sma(s, n)
    std = s.rolling(n).std(ddof=0)
    return mid + k * std, mid, mid - k * std


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with indicator columns. All are causal (use only current and past bars)."""
    out = df.copy()
    c = out["close"]
    out["ema20"] = ema(c, 20)
    out["ema50"] = ema(c, 50)
    out["ema200"] = ema(c, 200)
    out["sma20"] = sma(c, 20)
    out["rsi14"] = rsi(c, 14)
    out["macd"], out["macd_signal"], out["macd_hist"] = macd(c)
    out["atr14"] = atr(out, 14)
    out["atr_pct"] = out["atr14"] / c
    out["bb_upper"], out["bb_mid"], out["bb_lower"] = bollinger(c, 20)
    width = (out["bb_upper"] - out["bb_lower"]).replace(0, np.nan)
    out["bb_pos"] = ((c - out["bb_lower"]) / width).clip(0, 1)
    out["vol_sma20"] = sma(out["volume"], 20)
    out["vol_ratio"] = out["volume"] / out["vol_sma20"].replace(0, np.nan)
    out["ret_1"] = c.pct_change(1)
    out["ret_5"] = c.pct_change(5)
    out["ret_20"] = c.pct_change(20)
    out["hh_20"] = out["high"].rolling(20).max()
    out["ll_20"] = out["low"].rolling(20).min()
    return out


def _f(v, nd: int = 4):
    try:
        if v is None or pd.isna(v):
            return None
        return round(float(v), nd)
    except (TypeError, ValueError):
        return None


def snapshot(df: pd.DataFrame, symbol: str, timeframe: str, recent_bars: int = 30) -> dict:
    """Compact, LLM-friendly view of the latest state. Assumes add_indicators() was applied."""
    last = df.iloc[-1]
    price = float(last["close"])
    recent = df.tail(recent_bars)
    candles = [
        {
            "t": ts.strftime("%m-%d %H:%M"),
            "o": _f(r["open"]),
            "h": _f(r["high"]),
            "l": _f(r["low"]),
            "c": _f(r["close"]),
            "v": _f(r["volume"], 1),
        }
        for ts, r in recent.iterrows()
    ]
    if last["ema20"] > last["ema50"] > last["ema200"]:
        trend = "up"
    elif last["ema20"] < last["ema50"] < last["ema200"]:
        trend = "down"
    else:
        trend = "mixed"
    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "as_of": df.index[-1].strftime("%Y-%m-%d %H:%M UTC"),
        "price": price,
        "returns_pct": {
            "1_bar": _f(last["ret_1"] * 100, 2),
            "5_bars": _f(last["ret_5"] * 100, 2),
            "20_bars": _f(last["ret_20"] * 100, 2),
        },
        "trend_ema_20_50_200": trend,
        "ema": {"20": _f(last["ema20"]), "50": _f(last["ema50"]), "200": _f(last["ema200"])},
        "price_vs_ema200_pct": _f((price / last["ema200"] - 1) * 100, 2) if last["ema200"] else None,
        "rsi14": _f(last["rsi14"], 1),
        "macd": {"line": _f(last["macd"]), "signal": _f(last["macd_signal"]), "hist": _f(last["macd_hist"])},
        "atr_pct": _f(last["atr_pct"] * 100, 2),
        "bollinger_position_0_to_1": _f(last["bb_pos"], 2),
        "volume_vs_20bar_avg": _f(last["vol_ratio"], 2),
        "range_20_bars": {"high": _f(last["hh_20"]), "low": _f(last["ll_20"])},
        "recent_candles": candles,
    }
