# ai-trader

Your own AI trader: **Claude decides, rules protect, paper money by default.**

```
market data ──► indicators ──► Claude (structured decision) ──► risk manager ──► broker ──► journal
 ccxt / yfinance   pandas         claude-opus-5                  hard limits      paper | live   sqlite
```

- **Markets**: crypto via any [ccxt](https://github.com/ccxt/ccxt) exchange (default Binance, public data needs no account) and stocks/ETFs via Yahoo Finance. A symbol with `/` (`BTC/USDT`) is crypto, anything else (`AAPL`) is a stock.
- **Brain**: Claude gets a compact technical snapshot plus your account state and returns a typed `Decision` (buy / sell / hold, confidence, size, stop, target, reasoning, risks) via structured outputs.
- **Risk manager**: independent of the model. Caps position size, number of positions, daily loss, minimum confidence, and attaches a stop-loss and take-profit to every entry. Claude cannot override it.
- **Paper first**: simulated fills with fees and slippage, persisted in `state/`. Live trading exists but only arms when you set three things explicitly.
- **Backtester**: replays history bar by bar through the *same* risk manager and broker, with a free rule-based strategy as a benchmark.
- **ML strategy**: a LightGBM model you train yourself on indicator features, validated walk-forward and backtested only on data it never saw.

> This is a tool for learning and experimenting. Markets lose money. Nothing here is financial advice, and a language model reading 30 candles has no edge you should trust with money you cannot afford to lose. Run paper mode for weeks before you even think about live.

## Setup

```bash
cd ai-trader
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
copy .env.example .env          # then edit .env
```

Put your Anthropic key in `.env` as `ANTHROPIC_API_KEY=...` (or log in with the `ant` CLI and leave it empty). Everything else has sane defaults.

## Use it

```bash
# One-off analysis, no trade. Claude by default; --strategy rule needs no API key.
python -m ai_trader analyze BTC/USDT
python -m ai_trader analyze AAPL --timeframe 1d
python -m ai_trader analyze ETH/USDT --strategy rule

# Paper-trade the symbols from .env, one tick every LOOP_INTERVAL_SEC (default 1h)
python -m ai_trader run
python -m ai_trader run --once            # single tick, handy for cron / Task Scheduler
python -m ai_trader run --strategy rule   # free benchmark bot

# Backtest (rule strategy is free; Claude costs one API call per decision)
python -m ai_trader backtest BTC/USDT --days 180 --timeframe 4h --trades
python -m ai_trader backtest BTC/USDT --days 60 --timeframe 4h --strategy claude --every 6

# ML strategy: train, then backtest on the held-out last 90 days, then paper-trade
python -m ai_trader train BTC/USDT --timeframe 1h --days 730
python -m ai_trader backtest BTC/USDT --timeframe 1h --days 730 --strategy ml
python -m ai_trader run --strategy ml

# Portfolio, open positions, recent fills and decisions
python -m ai_trader status
python -m ai_trader history -n 30
python -m ai_trader reset                 # wipe the paper portfolio
```

## Configuration (`.env`)

| Key | Default | Meaning |
| --- | --- | --- |
| `CLAUDE_MODEL` / `CLAUDE_EFFORT` | `claude-opus-5` / `medium` | Model and reasoning depth (`low`..`max`). Higher effort costs more per decision. |
| `SYMBOLS` / `TIMEFRAME` | `BTC/USDT,ETH/USDT` / `1h` | What to trade and on which candles. |
| `EXCHANGE` | `binance` | Any ccxt exchange id. |
| `TRADING_MODE` | `paper` | `paper` or `live`. |
| `MAX_POSITION_PCT` | `0.20` | Max share of equity per position. Claude's `size_pct` scales *within* this. |
| `MAX_OPEN_POSITIONS` | `3` | |
| `MAX_DAILY_LOSS_PCT` | `0.03` | No new buys after losing this much in a UTC day. |
| `MIN_CONFIDENCE` | `0.60` | Decisions below this are ignored. |
| `DEFAULT_STOP_LOSS_PCT` / `DEFAULT_TAKE_PROFIT_PCT` | `0.03` / `0.06` | Used when the strategy does not propose its own (stop is at least 1.5x ATR). |
| `FEE_PCT` / `SLIPPAGE_PCT` | `0.001` / `0.0005` | Paper fill realism. |
| `LOOP_INTERVAL_SEC` | `3600` | Seconds between ticks in `run`. |

## ML strategy (LightGBM)

`train` fetches history, computes the same indicators the bot trades on, and learns one question:
*will the price in `--horizon` bars (default 12) be higher by more than a round trip of fees + slippage?*

- **Features** (`ml.make_features`): scale-free versions of the indicators: returns over 1/5/20 bars, volatility, RSI, MACD histogram / price, ATR %, Bollinger position, log volume ratio, distance between EMAs, position in the 20-bar range. All causal.
- **Validation**: walk-forward. Five expanding folds, each trained only on the past with a `horizon`-bar gap so no training label overlaps the test block. The last `--holdout-days` (default 90) are never used for training at all.
- **Report**: per fold and holdout: AUC (0.50 = coin flip), accuracy vs. the base rate, how many bars would have triggered a buy, their hit rate, and their average forward return vs. all bars. Look for AUC steadily above ~0.52 in *every* block and signal returns above the average. A single good fold is luck.
- **Model file**: `state/models/<SYMBOL>_<TF>.json`: LightGBM's text model plus metadata (training period, horizon, costs, out-of-sample start). No pickle, so loading it runs no code.
- **Backtest** with `--strategy ml` automatically starts at the model's out-of-sample start and refuses to run a model trained with `--holdout-days 0`, because that backtest would only measure memorisation.
- **Trading rules**: buy when P >= `MIN_CONFIDENCE`; sell when P <= 1 - `MIN_CONFIDENCE`, or when the position is older than the horizon and there is no fresh buy signal. Stops and targets come from the risk manager as for every strategy. `confidence` is the model's raw probability, so `MIN_CONFIDENCE` is the single knob for both the strategy and the risk gate.
- **Going to paper/live**: once you trust the numbers, retrain with `--holdout-days 0` so the model sees the most recent data, and set `LOOKBACK_BARS=1000` so EMA200 in live features is computed on as much history as in training. Retrain regularly (for example weekly); markets drift.

Expect the honest answer to often be "no edge": on hourly BTC the walk-forward AUC hovers around 0.50-0.57 and the holdout near 0.50. The value of the pipeline is that it tells you so before you lose money.

## Going live (read this twice)

Live mode sends real market orders through ccxt. It refuses to start unless **all** of these are in `.env`:

```
TRADING_MODE=live
LIVE_CONFIRM=I_UNDERSTAND_THE_RISKS
EXCHANGE_API_KEY=...
EXCHANGE_SECRET=...
```

Use an API key with **trade permission only** (no withdrawals), restricted to your IP, on a sub-account funded with an amount you are prepared to lose. Stocks are paper-only. Stops and targets are enforced by this process on each tick, not by exchange-side orders, so if the bot is not running, nothing protects the position.

## How a tick works

1. Fetch the last `LOOKBACK_BARS` candles for each symbol and compute EMA 20/50/200, RSI, MACD, ATR, Bollinger position, volume ratio.
2. If a position's stop or target was crossed by the latest bar, close it.
3. Build the snapshot and account context, ask the strategy for a `Decision`.
4. Risk manager: reject or shrink. Buys get a stop and target price attached.
5. Broker executes (paper or live). Portfolio is saved to `state/portfolio_<mode>.json`.
6. Every decision (approved or not, with reasoning) and every fill goes to `state/journal.sqlite`.

## Cost

One Claude call per symbol per tick, roughly 2-3k input tokens (the system prompt is cached) and a few hundred output tokens. Two symbols hourly on `claude-opus-5` at medium effort is on the order of a dollar or two per day. `CLAUDE_EFFORT=low` or a longer `TIMEFRAME` cuts it further.

## Layout

```
ai_trader/
  config.py     settings from .env (pydantic-settings)
  data.py       ccxt + yfinance candles
  indicators.py pandas indicators, LLM snapshot
  models.py     Decision (Claude's output schema), Position, Fill
  brain.py      ClaudeBrain: structured-output call, refusal/error handling
  strategy.py   RuleStrategy (free baseline), ClaudeStrategy
  ml.py         LightGBM features, labels, walk-forward, MLModel, MLStrategy
  risk.py       RiskManager: the limits Claude cannot bypass
  broker.py     Portfolio, PaperBroker, LiveBroker (ccxt)
  journal.py    sqlite log of decisions and fills
  engine.py     TradingEngine tick loop
  backtest.py   bar-by-bar replay with metrics
  cli.py        analyze | run | backtest | train | status | history | reset
tests/          pytest, no network, no API key needed
```

## Extending

- **New signals**: add columns in `indicators.add_indicators` and fields in `indicators.snapshot`. Claude sees whatever the snapshot contains. For the ML strategy, add a scale-free column in `ml.make_features` and retrain.
- **News / sentiment**: fetch it in the engine and pass it through `PortfolioContext.extra`; it is serialized into the prompt.
- **Another strategy**: implement `decide(symbol, df, ctx, timeframe) -> Decision` and register it in `cli.make_strategy`.
- **Shorting / futures**: the broker and risk manager are long-only by design; extending them is the place to start.
