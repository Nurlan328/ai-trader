"""CLI: analyze | run | backtest | train | status | history | reset"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from functools import partial

import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from . import data
from .backtest import run_backtest
from .broker import LiveBroker, PaperBroker, Portfolio
from .config import LIVE_CONFIRM_PHRASE, Settings, get_settings
from .engine import TradingEngine, build_context
from .indicators import add_indicators, snapshot
from .journal import Journal
from .strategy import ClaudeStrategy, RuleStrategy

console = Console()


def model_dir(s: Settings):
    return s.state_dir / "models"


def make_strategy(name: str, s: Settings):
    if name == "rule":
        return RuleStrategy()
    if name == "ml":
        from .ml import MLStrategy

        return MLStrategy(model_dir(s), threshold=s.min_confidence)
    from .brain import ClaudeBrain

    return ClaudeStrategy(ClaudeBrain(model=s.claude_model, effort=s.claude_effort))


def portfolio_path(s: Settings):
    return s.state_dir / f"portfolio_{s.trading_mode}.json"


def journal_path(s: Settings):
    return s.state_dir / "journal.sqlite"


def pct(x: float) -> str:
    return f"{x * 100:+.2f}%"


# --- commands ---------------------------------------------------------------

def cmd_analyze(args, s: Settings) -> None:
    strategy = make_strategy(args.strategy, s)
    df = add_indicators(data.fetch_ohlcv(args.symbol, args.timeframe or s.timeframe, s.lookback_bars, s.exchange))
    snap = snapshot(df, args.symbol, args.timeframe or s.timeframe)
    portfolio = Portfolio.load_or_create(portfolio_path(s), s.start_cash)
    price = snap["price"]
    ctx = build_context(portfolio, args.symbol, price, portfolio.equity({args.symbol: price}), s.max_open_positions)

    brief = {k: v for k, v in snap.items() if k != "recent_candles"}
    console.print(Panel(json.dumps(brief, indent=2), title=f"{args.symbol} snapshot", expand=False))
    with console.status(f"Asking {strategy.name} strategy..."):
        d = strategy.decide(args.symbol, df, ctx, args.timeframe or s.timeframe)
    color = {"buy": "green", "sell": "red", "hold": "yellow"}[d.action]
    body = f"[bold {color}]{d.action.upper()}[/]  confidence={d.confidence:.2f}  size={d.size_pct:.2f}"
    if d.stop_loss_pct:
        body += f"  stop={d.stop_loss_pct:.2%}"
    if d.take_profit_pct:
        body += f"  target={d.take_profit_pct:.2%}"
    body += f"\n\n{d.reasoning}"
    if d.key_risks:
        body += "\n\nRisks:\n" + "\n".join(f"  - {r}" for r in d.key_risks)
    console.print(Panel(body, title=f"Decision ({strategy.name})", expand=False))


def cmd_run(args, s: Settings) -> None:
    strategy = make_strategy(args.strategy, s)
    portfolio = Portfolio.load_or_create(portfolio_path(s), s.start_cash)
    journal = Journal(journal_path(s))

    if s.trading_mode == "live":
        if not s.live_enabled:
            console.print(
                f"[red]TRADING_MODE=live but live trading is not armed.[/] Set LIVE_CONFIRM={LIVE_CONFIRM_PHRASE}, "
                "EXCHANGE_API_KEY and EXCHANGE_SECRET in .env. Refusing to start."
            )
            sys.exit(2)
        if any(not data.is_crypto(sym) for sym in s.symbol_list):
            console.print("[red]Live mode supports crypto symbols only (via ccxt).[/]")
            sys.exit(2)
        ex = data.get_exchange(s.exchange, s.exchange_api_key, s.exchange_secret)
        broker = LiveBroker(portfolio, ex, fee_pct=s.fee_pct)
        console.print(Panel("[bold red]LIVE MODE: real orders will be sent to the exchange.[/]", expand=False))
    else:
        broker = PaperBroker(portfolio, s.fee_pct, s.slippage_pct)

    fetch = partial(data.fetch_ohlcv, timeframe=s.timeframe, limit=s.lookback_bars, exchange=s.exchange)
    engine = TradingEngine(s, strategy, broker, portfolio, journal, fetch, state_path=portfolio_path(s))
    if args.once:
        fills = engine.tick()
        console.print(f"tick done: {len(fills)} fill(s); equity={portfolio.equity(engine.last_prices):.2f}")
        return
    engine.run_forever(args.interval)


def cmd_backtest(args, s: Settings) -> None:
    tf = args.timeframe or s.timeframe
    strategy = make_strategy(args.strategy, s)
    if strategy.name == "claude":
        console.print(f"[yellow]Claude backtest: ~{args.days * 24 // (data.timeframe_seconds(tf) // 3600 or 1) // args.every} API calls; use --every to reduce.[/]")
    with console.status(f"Fetching {args.days}d of {tf} candles for {args.symbol}..."):
        df = data.fetch_history(args.symbol, tf, args.days, s.exchange)
    warmup = min(200, len(df) // 3)
    if strategy.name == "ml":
        warmup = max(warmup, ml_oos_start_bar(strategy, df, args.symbol, tf))
    with console.status(f"Backtesting {len(df) - warmup} bars with {strategy.name}..."):
        r = run_backtest(df, strategy, s, args.symbol, tf, warmup=warmup, every=args.every)

    t = Table(title=f"Backtest {r.symbol} {tf}  {r.start} -> {r.end}  ({r.bars} bars, strategy={r.strategy})")
    t.add_column("Metric")
    t.add_column("Value", justify="right")
    rows = [
        ("Final equity", f"{r.final_equity:,.2f} (start {r.start_cash:,.0f})"),
        ("Total return", pct(r.total_return)),
        ("Buy & hold", pct(r.buy_hold_return)),
        ("Max drawdown", pct(r.max_drawdown)),
        ("Sharpe (annualized)", f"{r.sharpe:.2f}"),
        ("Round trips", str(r.n_round_trips)),
        ("Win rate", pct(r.win_rate)),
        ("Profit factor", f"{r.profit_factor:.2f}"),
        ("Fees paid", f"{r.fees_paid:,.2f}"),
    ]
    for k, v in rows:
        t.add_row(k, v)
    console.print(t)

    if args.trades:
        ft = Table(title="Fills")
        for c in ("time", "side", "qty", "price", "pnl", "reason"):
            ft.add_column(c)
        for f in r.fills:
            ft.add_row(f.ts[:16], f.side, f"{f.qty:.6f}", f"{f.price:,.4f}", f"{f.pnl:+.2f}", f.reason)
        console.print(ft)


def ml_oos_start_bar(strategy, df: pd.DataFrame, symbol: str, tf: str) -> int:
    """First bar the model never trained on. Backtesting on training bars would only show memorisation."""
    m = strategy.model_for(symbol, tf)
    if m is None:
        console.print(f"[red]No ML model for {symbol} {tf}.[/] Train one: python -m ai_trader train {symbol} --timeframe {tf}")
        sys.exit(2)
    if m.oos_start is None:
        console.print("[red]This model was trained on all history (--holdout-days 0), so any backtest is in-sample.[/] "
                      "Retrain with --holdout-days N and backtest those N days.")
        sys.exit(2)
    start = int(df.index.searchsorted(pd.Timestamp(m.oos_start)))
    if start >= len(df) - 2:
        console.print(f"[red]No bars after the model's out-of-sample start {m.oos_start[:16]}.[/]")
        sys.exit(2)
    console.print(f"[yellow]ML backtest runs out-of-sample only: from {df.index[start]:%Y-%m-%d %H:%M} ({len(df) - start} bars).[/]")
    return start


def cmd_train(args, s: Settings) -> None:
    from .ml import model_path, train

    tf = args.timeframe or s.timeframe
    with console.status(f"Fetching {args.days}d of {tf} candles for {args.symbol}..."):
        raw = data.fetch_history(args.symbol, tf, args.days, s.exchange)
    holdout_start = raw.index[-1] - pd.Timedelta(days=args.holdout_days) if args.holdout_days > 0 else None
    cost = 2 * (s.fee_pct + s.slippage_pct)  # round trip
    with console.status(f"Walk-forward training on {len(raw)} bars..."):
        m, rep = train(raw, args.symbol, tf, horizon=args.horizon, cost=cost, holdout_start=holdout_start,
                       threshold=s.min_confidence, n_splits=args.splits)

    t = Table(title=f"{args.symbol} {tf}: P(+{args.horizon} bars > {cost:.2%} costs), signal = P >= {s.min_confidence:.2f}")
    for c in ("block", "period", "train", "test", "AUC", "acc", "base rate", "signals", "hit rate", "fwd ret signals", "fwd ret all"):
        t.add_column(c, justify="right")

    def row(name: str, f: dict, n_train: str) -> None:
        t.add_row(name, f"{f['start']} -> {f['end']}", n_train, str(f["n"]), f"{f['auc']:.3f}", f"{f['accuracy']:.3f}",
                  f"{f['base_rate']:.3f}", str(f["n_signals"]), f"{f['hit_rate']:.3f}", pct(f["fwd_ret_signals"]), pct(f["fwd_ret_all"]))

    for f in rep.folds:
        row(f"fold {f['fold']}", f, str(f["n_train"]))
    if rep.holdout:
        t.add_section()
        row("holdout", rep.holdout, str(m.meta["n_train"]))
    console.print(t)
    console.print("AUC 0.50 = coin flip. What counts: AUC steadily above ~0.52 in every block, and 'fwd ret signals' above 'fwd ret all'.")

    imp = Table(title="Feature importance (share of gain)")
    imp.add_column("feature")
    imp.add_column("share", justify="right")
    for name, share in rep.importance[:10]:
        imp.add_row(name, f"{share:.1%}")
    console.print(imp)

    path = model_path(model_dir(s), args.symbol, tf)
    m.save(path)
    console.print(f"saved {path}")
    if m.oos_start:
        console.print(f"next: python -m ai_trader backtest {args.symbol} --timeframe {tf} --days {args.days} --strategy ml")


def cmd_status(args, s: Settings) -> None:
    portfolio = Portfolio.load_or_create(portfolio_path(s), s.start_cash)
    prices: dict[str, float] = {}
    for sym in portfolio.positions:
        try:
            prices[sym] = data.latest_price(sym, s.exchange)
        except Exception as e:  # keep status usable offline
            console.print(f"[yellow]price fetch failed for {sym}: {e}[/]")
    equity = portfolio.equity(prices)
    console.print(
        Panel(
            f"mode={s.trading_mode}  cash={portfolio.cash:,.2f}  equity={equity:,.2f}  "
            f"return={pct(equity / portfolio.start_cash - 1)}  realized={portfolio.realized_pnl:+,.2f}  "
            f"fees={portfolio.fees_paid:,.2f}  trades={portfolio.n_trades}",
            title="Portfolio",
            expand=False,
        )
    )
    if portfolio.positions:
        t = Table(title="Open positions")
        for c in ("symbol", "qty", "entry", "now", "unrealized", "stop", "target", "opened"):
            t.add_column(c)
        for sym, p in portfolio.positions.items():
            now = prices.get(sym, p.entry_price)
            t.add_row(sym, f"{p.qty:.6f}", f"{p.entry_price:,.4f}", f"{now:,.4f}", pct(p.unrealized_pct(now)),
                      f"{p.stop_loss:,.4f}" if p.stop_loss else "-", f"{p.take_profit:,.4f}" if p.take_profit else "-", p.opened_at[:16])
        console.print(t)


def cmd_history(args, s: Settings) -> None:
    journal = Journal(journal_path(s))
    ft = Table(title=f"Last {args.n} fills")
    for c in ("time", "symbol", "side", "qty", "price", "pnl", "reason"):
        ft.add_column(c)
    for f in journal.recent_fills(args.n):
        ft.add_row(f["ts"][:16], f["symbol"], f["side"], f"{f['qty']:.6f}", f"{f['price']:,.4f}", f"{f['pnl']:+.2f}", f["reason"])
    console.print(ft)
    dt = Table(title=f"Last {args.n} decisions")
    for c in ("time", "symbol", "action", "conf", "exec", "risk", "reasoning"):
        dt.add_column(c)
    for d in journal.recent_decisions(args.n):
        dt.add_row(d["ts"][:16], d["symbol"], d["action"], f"{d['confidence']:.2f}", "yes" if d["approved"] else "no", d["risk_reason"], d["reasoning"][:80])
    console.print(dt)


def cmd_reset(args, s: Settings) -> None:
    path = portfolio_path(s)
    if s.trading_mode == "live":
        console.print("[red]Refusing to reset the live portfolio file automatically; delete it by hand if you really mean it.[/]")
        sys.exit(2)
    if path.exists():
        path.unlink()
    console.print(f"paper portfolio reset to {s.start_cash:,.0f} (removed {path})")


# --- main -------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="ai-trader", description="Claude-powered trading bot (paper by default).")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("analyze", help="one-off decision for a symbol, no trade")
    a.add_argument("symbol")
    a.add_argument("--strategy", choices=["claude", "rule", "ml"], default="claude")
    a.add_argument("--timeframe")

    r = sub.add_parser("run", help="trading loop (paper unless live is armed)")
    r.add_argument("--strategy", choices=["claude", "rule", "ml"], default="claude")
    r.add_argument("--once", action="store_true", help="single tick then exit")
    r.add_argument("--interval", type=int, help="seconds between ticks")

    b = sub.add_parser("backtest", help="replay history bar by bar")
    b.add_argument("symbol")
    b.add_argument("--strategy", choices=["claude", "rule", "ml"], default="rule")
    b.add_argument("--days", type=int, default=180)
    b.add_argument("--timeframe")
    b.add_argument("--every", type=int, default=1, help="call strategy every N bars")
    b.add_argument("--trades", action="store_true", help="list fills")

    tr = sub.add_parser("train", help="train the LightGBM model for --strategy ml")
    tr.add_argument("symbol")
    tr.add_argument("--timeframe")
    tr.add_argument("--days", type=int, default=730, help="history to fetch")
    tr.add_argument("--horizon", type=int, default=12, help="bars ahead the model predicts")
    tr.add_argument("--holdout-days", type=int, default=90, help="most recent days kept out of training for an honest backtest; 0 = train on everything")
    tr.add_argument("--splits", type=int, default=5, help="walk-forward folds")

    sub.add_parser("status", help="portfolio and open positions")
    h = sub.add_parser("history", help="recent fills and decisions")
    h.add_argument("-n", type=int, default=20)
    sub.add_parser("reset", help="reset the paper portfolio")

    args = p.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("httpx", "httpcore", "urllib3", "ccxt", "yfinance", "peewee"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    s = get_settings()
    {"analyze": cmd_analyze, "run": cmd_run, "backtest": cmd_backtest, "train": cmd_train, "status": cmd_status, "history": cmd_history, "reset": cmd_reset}[args.cmd](args, s)


if __name__ == "__main__":
    main()
