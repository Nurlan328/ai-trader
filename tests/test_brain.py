"""Brain tests with a fake Anthropic client: verifies request shape and failure handling, no network."""
from types import SimpleNamespace

import anthropic
import httpx2 as httpx
import pytest

from ai_trader.brain import SYSTEM_PROMPT, ClaudeBrain
from ai_trader.models import Decision, PortfolioContext

CTX = PortfolioContext(cash=1000, equity=1000, open_positions=0, max_positions=3, daily_pnl_pct=0.0)
SNAP = {"symbol": "BTC/USDT", "price": 100.0}


def fake_response(parsed, stop_reason="end_turn", stop_details=None):
    return SimpleNamespace(
        parsed_output=parsed,
        stop_reason=stop_reason,
        stop_details=stop_details,
        usage=SimpleNamespace(input_tokens=10, cache_read_input_tokens=0, output_tokens=5),
    )


class FakeClient:
    def __init__(self, response=None, error=None):
        self.calls = []
        self.messages = self

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self._error, Exception):
            raise self._error
        return self._response

    def with_response(self, response):
        self._response, self._error = response, None
        return self

    def with_error(self, error):
        self._response, self._error = None, error
        return self


def test_request_shape_and_normalization():
    raw = Decision(action="buy", confidence=1.7, size_pct=0.5, stop_loss_pct=0.03, take_profit_pct=0.08, reasoning="ok")
    client = FakeClient().with_response(fake_response(raw))
    d = ClaudeBrain(model="claude-opus-5", effort="low", client=client).decide(SNAP, CTX)

    assert d.action == "buy" and d.confidence == 1.0  # clamped
    call = client.calls[0]
    assert call["model"] == "claude-opus-5"
    assert call["output_format"] is Decision
    assert call["thinking"] == {"type": "adaptive"}
    assert call["output_config"] == {"effort": "low"}
    assert call["system"][0]["text"] == SYSTEM_PROMPT
    assert call["system"][0]["cache_control"] == {"type": "ephemeral"}
    user = call["messages"][0]["content"]
    assert '"symbol":"BTC/USDT"' in user and '"cash":1000' in user


def test_refusal_becomes_hold():
    client = FakeClient().with_response(fake_response(None, stop_reason="refusal", stop_details=SimpleNamespace(explanation="nope")))
    d = ClaudeBrain(client=client).decide(SNAP, CTX)
    assert d.action == "hold" and "declined" in d.reasoning


def test_unparsable_becomes_hold():
    client = FakeClient().with_response(fake_response(None, stop_reason="max_tokens"))
    assert ClaudeBrain(client=client).decide(SNAP, CTX).action == "hold"


@pytest.mark.parametrize(
    "status,cls",
    [(429, anthropic.RateLimitError), (500, anthropic.InternalServerError), (400, anthropic.BadRequestError)],
)
def test_api_errors_become_hold(status, cls):
    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    resp = httpx.Response(status, request=req, headers={"retry-after": "7"})
    err = cls("boom", response=resp, body=None)
    d = ClaudeBrain(client=FakeClient().with_error(err)).decide(SNAP, CTX)
    assert d.action == "hold"


def test_connection_error_becomes_hold():
    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    err = anthropic.APIConnectionError(request=req)
    assert ClaudeBrain(client=FakeClient().with_error(err)).decide(SNAP, CTX).action == "hold"
