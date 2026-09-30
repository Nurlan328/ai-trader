"""Market data. Crypto via ccxt (public endpoints, no key needed), stocks via yfinance."""
from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

import pandas as pd

COLUMNS = ["open", "high", "low", "close", "volume"]

_TF_SECONDS = {"m": 60, "h": 3600, "d": 86400, "w": 604800}


def is_crypto(symbol: str) -> bool:
    return "/" in symbol


def timeframe_seconds(tf: str) -> int:
    unit = tf[-1]
    if unit not in _TF_SECONDS:
        raise ValueError(f"Unsupported timeframe: {tf}")
    return int(tf[:-1]) * _TF_SECONDS[unit]


def bars_per_year(tf: str) -> float:
    return 365 * 86400 / timeframe_seconds(tf)


def fetch_ohlcv(symbol: str, timeframe: str = "1h", limit: int = 300, exchange: str = "binance") -> pd.DataFrame:
    """Most recent `limit` candles, oldest first, UTC DatetimeIndex."""
    if is_crypto(symbol):
        return _ccxt_recent(symbol, timeframe, limit, exchange)
    return _yf_history(symbol, timeframe, days=_days_for(timeframe, limit)).tail(limit)


def fetch_history(symbol: str, timeframe: str, days: int, exchange: str = "binance") -> pd.DataFrame:
    """Up to `days` of candles for backtesting (paginates ccxt)."""
    if is_crypto(symbol):
        return _ccxt_since(symbol, timeframe, days, exchange)
    return _yf_history(symbol, timeframe, days)


def latest_price(symbol: str, exchange: str = "binance") -> float:
    if is_crypto(symbol):
        ex = get_exchange(exchange)
        return float(ex.fetch_ticker(symbol)["last"])
    import yfinance as yf

    info = yf.Ticker(symbol).fast_info
    return float(info["last_price"])


# --- ccxt -------------------------------------------------------------------

_EX_CACHE: dict[str, object] = {}


def get_exchange(exchange_id: str, api_key: str = "", secret: str = ""):
    import ccxt

    key = f"{exchange_id}:{bool(api_key)}"
    if key not in _EX_CACHE:
        cls = getattr(ccxt, exchange_id)
        params: dict = {"enableRateLimit": True}
        if api_key:
            params.update({"apiKey": api_key, "secret": secret})
        _EX_CACHE[key] = cls(params)
    return _EX_CACHE[key]


def _to_frame(rows: list) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["ts", *COLUMNS])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.drop_duplicates("ts").set_index("ts").sort_index()
    return df.astype(float)


def _ccxt_recent(symbol: str, timeframe: str, limit: int, exchange_id: str) -> pd.DataFrame:
    ex = get_exchange(exchange_id)
    rows = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    return _to_frame(rows)


def _ccxt_since(symbol: str, timeframe: str, days: int, exchange_id: str) -> pd.DataFrame:
    ex = get_exchange(exchange_id)
    step_ms = timeframe_seconds(timeframe) * 1000
    since = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)
    rows: list = []
    while True:
        batch = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=1000)
        if not batch:
            break
        rows.extend(batch)
        last = batch[-1][0]
        if len(batch) < 2 or last + step_ms > time.time() * 1000:
            break
        since = last + step_ms
    return _to_frame(rows)


# --- yfinance ---------------------------------------------------------------

_YF_INTERVALS = {"1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m", "1h": "1h", "1d": "1d", "1w": "1wk"}


def _days_for(timeframe: str, limit: int) -> int:
    # Slack for weekends and market hours: stocks trade ~6.5h of 24 on ~5 of 7 days.
    seconds = timeframe_seconds(timeframe) * limit
    factor = 1.0 if timeframe.endswith("w") else 4.0
    return max(2, int(seconds / 86400 * factor) + 2)


def _yf_history(symbol: str, timeframe: str, days: int) -> pd.DataFrame:
    import yfinance as yf

    interval = _YF_INTERVALS.get(timeframe)
    if interval is None:
        raise ValueError(f"Timeframe {timeframe} not supported for stocks; use one of {list(_YF_INTERVALS)}")
    # Yahoo caps intraday history: 1h -> 730 days, under 1h -> 60 days.
    if interval == "1h":
        days = min(days, 729)
    elif interval.endswith("m"):
        days = min(days, 59)
    raw = yf.Ticker(symbol).history(period=f"{days}d", interval=interval, auto_adjust=True)
    if raw.empty:
        raise RuntimeError(f"No data returned for {symbol} ({interval}, {days}d)")
    df = raw.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
    df = df[COLUMNS].astype(float)
    df.index = pd.to_datetime(df.index, utc=True)
    df.index.name = "ts"
    return df
