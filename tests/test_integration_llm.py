"""
Real, network-hitting integration tests against the live DeepSeek API.

These are the ONE place in the suite allowed to make a real call. Everywhere
else, `llm_client.call_llm` / `openai.OpenAI` are mocked (see
tests/conftest.py's fake_openai_client fixture and docs/TESTING_STRATEGY.md).

Run these explicitly, on request, since they cost real API tokens and
require a real key:

    DEEPSEEK_API_KEY=sk-... pytest -m integration -v

They are skipped automatically (not failed) when DEEPSEEK_API_KEY isn't set,
so a plain `pytest` run of the full suite never needs a key and never spends
money.
"""
import json
import os

import pytest

from poetry_diacritization import config
from poetry_diacritization.llm_client import (
    SlidingWindowRateLimiter,
    call_llm,
    setup_client,
)

pytestmark = pytest.mark.integration

requires_api_key = pytest.mark.skipif(
    not os.getenv("DEEPSEEK_API_KEY"),
    reason="DEEPSEEK_API_KEY not set — skipping real-API integration test",
)


@pytest.fixture
def real_client():
    return setup_client()


@requires_api_key
def test_real_call_llm_returns_parseable_json(real_client):
    """
    Sends one small, real batch (2 verses) to DeepSeek with thinking
    disabled (the documented default) and checks that:
      - no error came back
      - the response is valid JSON
      - the response is shaped like a list of {id, sadr, ajuz} objects
        covering the same verse_ids we sent
    This does NOT check diacritization *quality* — that's a job for
    validate.py + pyarud against real data, not this smoke test.
    """
    rate_limiter = SlidingWindowRateLimiter(max_requests=config.REQUESTS_PER_MINUTE)
    verses = [
        {"id": "test_001", "sadr": "قفا نبك من ذكرى حبيب ومنزل", "ajuz": "بسقط اللوى بين الدخول فحومل"},
        {"id": "test_002", "sadr": "فتوضح فالمقراة لم يعف رسمها", "ajuz": "لما نسجتها من جنوب وشمأل"},
    ]

    content, reasoning, error, usage = call_llm(
        real_client,
        verses,
        rate_limiter,
        thinking_enabled=False,
    )

    assert error is None, f"DeepSeek call failed: {error}"
    assert content is not None
    assert reasoning is None  # thinking was off, so no reasoning trace expected

    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[len("json"):].strip()

    parsed = json.loads(cleaned)
    if isinstance(parsed, dict):
        list_values = [v for v in parsed.values() if isinstance(v, list)]
        assert list_values, f"response had no array of verses: {parsed}"
        parsed = list_values[0]

    assert isinstance(parsed, list)
    returned_ids = {str(item.get("id") or item.get("verse_id")) for item in parsed}
    assert returned_ids == {"test_001", "test_002"}
    for item in parsed:
        assert item.get("sadr")
        assert item.get("ajuz")


@requires_api_key
def test_real_call_llm_with_thinking_enabled_returns_reasoning(real_client):
    """
    Same smoke test, but with thinking mode on — confirms
    reasoning_content actually comes back non-empty when we ask DeepSeek
    to think, since generate.py's whole reasoning-artifact feature is only
    meaningful if this is true.
    """
    rate_limiter = SlidingWindowRateLimiter(max_requests=config.REQUESTS_PER_MINUTE)
    verses = [
        {"id": "test_001", "sadr": "قفا نبك من ذكرى حبيب ومنزل", "ajuz": "بسقط اللوى بين الدخول فحومل"},
    ]

    content, reasoning, error, usage = call_llm(
        real_client,
        verses,
        rate_limiter,
        thinking_enabled=True,
        reasoning_effort="high",
    )

    assert error is None, f"DeepSeek call failed: {error}"
    assert content is not None
    assert reasoning is not None and len(reasoning.strip()) > 0


@requires_api_key
def test_real_setup_client_authenticates(real_client):
    # setup_client() itself calls client.models.list() and would already
    # have raised/exited on a bad key by the time we get the fixture, so
    # reaching this point at all is the assertion. This test exists mainly
    # as a fast, cheap (no chat completion) canary for auth/connectivity
    # issues, separate from the more expensive generation tests above.
    assert real_client is not None
