"""Claude as the decision engine. One structured-output call per symbol per tick."""
from __future__ import annotations

import json
import logging
from dataclasses import asdict

import anthropic

from .models import Decision, PortfolioContext

log = logging.getLogger(__name__)

# Keep this stable: it is the cached prefix. Anything that changes per request goes in the user message.
SYSTEM_PROMPT = """You are a disciplined, risk-aware trading analyst for a single trading account.

Each request gives you one instrument: a technical snapshot (price, trend, momentum, volatility,
volume, recent candles) and the account state (cash, equity, open positions, today's P&L, and the
current position in this instrument if any). You decide ONE action for this instrument right now:

- "buy": open a new long position (only if there is no position in this instrument).
- "sell": close the existing position, fully or partially (size_pct = fraction of the position to close).
- "hold": do nothing.

Rules you must follow:
1. Default to "hold". Only act when several independent signals agree and the reward/risk is favorable.
2. Never buy against a clear downtrend, into extreme overbought readings, or when volatility is spiking
   without a reason you can articulate from the data.
3. confidence is your honest probability (0..1) that the action is right. Anything below 0.6 should be a hold.
4. size_pct (0..1): for buys, the fraction of the maximum allowed position size to deploy; for sells, the
   fraction of the position to close. Scale it with confidence and inversely with volatility (atr_pct).
5. Always propose stop_loss_pct and take_profit_pct for buys as positive fractions of the entry price
   (e.g. 0.03 = 3%). Stops should sit beyond recent structure and at least ~1.5x ATR.
6. reasoning: 2-4 short sentences citing the specific numbers that drove the decision.
7. key_risks: 1-3 short bullet strings describing what would invalidate the idea.
8. A separate risk manager will cap sizes and enforce limits; you do not need to re-check them, but respect
   the spirit: no new buys when today's loss is already large, and prefer closing losers to opening new risk.

You cannot see news or order books. Be honest about that uncertainty."""


class ClaudeBrain:
    def __init__(self, model: str = "claude-opus-5", effort: str = "medium", client: anthropic.Anthropic | None = None):
        self.model = model
        self.effort = effort
        self.client = client or anthropic.Anthropic()

    def decide(self, snapshot: dict, ctx: PortfolioContext) -> Decision:
        user_msg = (
            "MARKET SNAPSHOT (JSON):\n"
            + json.dumps(snapshot, separators=(",", ":"))
            + "\n\nACCOUNT STATE (JSON):\n"
            + json.dumps(asdict(ctx), separators=(",", ":"), default=str)
            + "\n\nDecide the action for this instrument now."
        )
        try:
            response = self.client.messages.parse(
                model=self.model,
                max_tokens=4096,
                system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": user_msg}],
                thinking={"type": "adaptive"},
                output_config={"effort": self.effort},
                output_format=Decision,
            )
        except anthropic.RateLimitError as e:
            retry_after = e.response.headers.get("retry-after", "?")
            log.warning("Rate limited by Claude API (retry-after=%s); holding.", retry_after)
            return Decision.hold("Claude API rate limited")
        except anthropic.APIStatusError as e:
            log.error("Claude API error %s: %s", e.status_code, e.message)
            return Decision.hold(f"Claude API error {e.status_code}")
        except anthropic.APIConnectionError as e:
            log.error("Claude API connection error: %s", e)
            return Decision.hold("Claude API unreachable")

        if response.stop_reason == "refusal":
            detail = response.stop_details.explanation if response.stop_details else ""
            log.warning("Claude declined the request: %s", detail)
            return Decision.hold("Claude declined to decide")
        if response.stop_reason == "max_tokens" or response.parsed_output is None:
            log.warning("Claude returned no parsable decision (stop_reason=%s)", response.stop_reason)
            return Decision.hold("No parsable decision from Claude")

        usage = response.usage
        log.info(
            "Claude decision for %s: %s (conf=%.2f) tokens in=%s cached=%s out=%s",
            snapshot.get("symbol"),
            response.parsed_output.action,
            response.parsed_output.confidence,
            usage.input_tokens,
            usage.cache_read_input_tokens,
            usage.output_tokens,
        )
        return response.parsed_output.normalized()
