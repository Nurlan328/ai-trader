from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

LIVE_CONFIRM_PHRASE = "I_UNDERSTAND_THE_RISKS"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Claude
    claude_model: str = "claude-opus-5"
    claude_effort: Literal["low", "medium", "high", "xhigh", "max"] = "medium"

    # Market
    exchange: str = "binance"
    symbols: str = "BTC/USDT,ETH/USDT"
    timeframe: str = "1h"
    lookback_bars: int = 300

    # Mode
    trading_mode: Literal["paper", "live"] = "paper"
    live_confirm: str = ""
    exchange_api_key: str = ""
    exchange_secret: str = ""

    # Portfolio & risk
    start_cash: float = 10_000.0
    max_position_pct: float = 0.20
    max_open_positions: int = 3
    max_daily_loss_pct: float = 0.03
    min_confidence: float = 0.60
    min_notional: float = 10.0
    fee_pct: float = 0.001
    slippage_pct: float = 0.0005
    default_stop_loss_pct: float = 0.03
    default_take_profit_pct: float = 0.06

    state_dir: Path = Path("state")
    loop_interval_sec: int = 3600

    @property
    def symbol_list(self) -> list[str]:
        return [s.strip() for s in self.symbols.split(",") if s.strip()]

    @property
    def live_enabled(self) -> bool:
        return (
            self.trading_mode == "live"
            and self.live_confirm == LIVE_CONFIRM_PHRASE
            and bool(self.exchange_api_key)
            and bool(self.exchange_secret)
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
