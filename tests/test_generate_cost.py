"""
Tests for the cost-tracking columns generate.py writes to the registry
(see registry.REGISTRY_COLUMNS and generate._cost_fields_for_batch).

These live separately from test_generate.py so the pre-existing file (and
its pre-existing assertions) stays untouched — this is purely additive
coverage for the new feature.
"""
import threading

import pytest

from poetry_diacritization import config, generate, pricing
from poetry_diacritization.generate import _cost_fields_for_batch, _process_one_batch

SAMPLE_PRICING = {
    "deepseek-v4-flash": {
        "currency": "USD",
        "unit_size": 1_000_000,
        "input_cache_miss_price": 0.14,
        "input_cache_hit_price": 0.0028,
        "output_price": 0.28,
    },
}


def _batch_for(verse_ids, poem_no=1):
    return {
        "poem_no": poem_no,
        "meter": "wafer",
        "verses": [{"id": vid, "sadr": "s", "ajuz": "a"} for vid in verse_ids],
    }


# ---------------------------------------------------------------------------
# _cost_fields_for_batch — the per-verse-share math, in isolation
# ---------------------------------------------------------------------------


def test_cost_fields_for_batch_splits_usage_evenly_across_verses(monkeypatch):
    monkeypatch.setattr(pricing, "load_pricing_config", lambda: SAMPLE_PRICING)
    usage = {
        "prompt_tokens": 400,
        "completion_tokens": 200,
        "cache_hit_tokens": 100,
        "cache_miss_tokens": 300,
    }

    fields = _cost_fields_for_batch("deepseek-v4-flash", usage, n=4)

    assert fields["input_tokens_cache_hit"] == 25
    assert fields["input_tokens_cache_miss"] == 75
    assert fields["output_tokens"] == 50
    expected_total_cost = (100 / 1_000_000) * 0.0028 + (300 / 1_000_000) * 0.14 + (
        200 / 1_000_000
    ) * 0.28
    assert fields["call_cost_usd"] == pytest.approx(expected_total_cost / 4)


def test_cost_fields_for_batch_none_usage_returns_none():
    assert _cost_fields_for_batch("deepseek-v4-flash", None, n=3) is None


def test_cost_fields_for_batch_zero_verses_returns_none():
    usage = {"prompt_tokens": 1, "completion_tokens": 1, "cache_hit_tokens": 1, "cache_miss_tokens": 0}
    assert _cost_fields_for_batch("deepseek-v4-flash", usage, n=0) is None


def test_cost_fields_for_batch_keeps_token_shares_when_model_unpriced(monkeypatch):
    monkeypatch.setattr(pricing, "load_pricing_config", lambda: SAMPLE_PRICING)
    usage = {
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "cache_hit_tokens": 40,
        "cache_miss_tokens": 60,
    }

    fields = _cost_fields_for_batch("some-unpriced-model", usage, n=2)

    # Cost can't be computed, but the token counts are still worth keeping.
    assert fields["call_cost_usd"] is None
    assert fields["input_tokens_cache_hit"] == 20
    assert fields["input_tokens_cache_miss"] == 30
    assert fields["output_tokens"] == 25


# ---------------------------------------------------------------------------
# _process_one_batch — end to end through the registry
# ---------------------------------------------------------------------------


def test_process_one_batch_populates_cost_columns_on_success(monkeypatch, make_registry_df):
    monkeypatch.setattr(pricing, "load_pricing_config", lambda: SAMPLE_PRICING)
    df = make_registry_df(
        [
            {"verse_id": "1_001", "poem_no": 1, "status": "pending"},
            {"verse_id": "1_002", "poem_no": 1, "status": "pending"},
        ]
    )
    batch = _batch_for(["1_001", "1_002"])
    usage = {
        "prompt_tokens": 200,
        "completion_tokens": 100,
        "cache_hit_tokens": 50,
        "cache_miss_tokens": 150,
    }
    monkeypatch.setattr(generate, "call_llm", lambda *a, **k: ("[]", None, None, usage))
    monkeypatch.setattr(generate, "save_registry", lambda d: None)

    _process_one_batch(
        client=None,
        batch=batch,
        rate_limiter=None,
        df=df,
        lock=threading.Lock(),
        save_reasoning=False,
        model="deepseek-v4-flash",
        thinking_enabled=False,
        reasoning_effort="high",
    )

    for verse_id in ["1_001", "1_002"]:
        assert df.loc[verse_id, "input_tokens_cache_hit"] == 25
        assert df.loc[verse_id, "input_tokens_cache_miss"] == 75
        assert df.loc[verse_id, "output_tokens"] == 50
        assert df.loc[verse_id, "call_cost_usd"] > 0
        # First attempt: cumulative equals this call's share exactly.
        assert df.loc[verse_id, "cumulative_cost_usd"] == pytest.approx(
            df.loc[verse_id, "call_cost_usd"]
        )

    # Splitting one call's cost across the batch, then summing back up,
    # must reproduce the whole call's cost (the whole point of dividing
    # by n rather than duplicating the raw total).
    whole_call_cost = pricing.calculate_cost(
        "deepseek-v4-flash", 50, 150, 100, pricing=SAMPLE_PRICING
    )
    assert df["cumulative_cost_usd"].sum() == pytest.approx(whole_call_cost)


def test_process_one_batch_accumulates_cost_across_repeated_passes(monkeypatch, make_registry_df):
    monkeypatch.setattr(pricing, "load_pricing_config", lambda: SAMPLE_PRICING)
    df = make_registry_df(
        [{"verse_id": "1_001", "poem_no": 1, "status": "failed_prosody"}]
    )
    batch = _batch_for(["1_001"])
    usage = {
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "cache_hit_tokens": 0,
        "cache_miss_tokens": 100,
    }
    monkeypatch.setattr(generate, "call_llm", lambda *a, **k: ("[]", None, None, usage))
    monkeypatch.setattr(generate, "save_registry", lambda d: None)

    kwargs = dict(
        client=None,
        rate_limiter=None,
        df=df,
        lock=threading.Lock(),
        save_reasoning=False,
        model="deepseek-v4-flash",
        thinking_enabled=False,
        reasoning_effort="high",
    )
    _process_one_batch(batch=batch, **kwargs)
    first_cost = df.loc["1_001", "cumulative_cost_usd"]
    _process_one_batch(batch=batch, **kwargs)
    second_cost = df.loc["1_001", "cumulative_cost_usd"]

    assert second_cost == pytest.approx(first_cost * 2)


def test_process_one_batch_leaves_cost_columns_none_when_usage_missing(monkeypatch, make_registry_df):
    df = make_registry_df([{"verse_id": "1_001", "poem_no": 1, "status": "pending"}])
    batch = _batch_for(["1_001"])
    # usage=None simulates an API response with no usage block at all.
    monkeypatch.setattr(generate, "call_llm", lambda *a, **k: ("[]", None, None, None))
    monkeypatch.setattr(generate, "save_registry", lambda d: None)

    _process_one_batch(
        client=None,
        batch=batch,
        rate_limiter=None,
        df=df,
        lock=threading.Lock(),
        save_reasoning=False,
        model="deepseek-v4-flash",
        thinking_enabled=False,
        reasoning_effort="high",
    )

    # No crash, no cost — but the normal, non-cost fields still updated.
    assert df.loc["1_001", "status"] == "awaiting_validation"
    assert df.loc["1_001", "call_cost_usd"] is None
    assert df.loc["1_001", "cumulative_cost_usd"] == 0.0


def test_process_one_batch_never_raises_when_pricing_config_absent(monkeypatch, make_registry_df):
    # Simulates a user who deleted pricing_config.py entirely.
    monkeypatch.setattr(pricing, "load_pricing_config", lambda: {})
    df = make_registry_df([{"verse_id": "1_001", "poem_no": 1, "status": "pending"}])
    batch = _batch_for(["1_001"])
    usage = {
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "cache_hit_tokens": 0,
        "cache_miss_tokens": 100,
    }
    monkeypatch.setattr(generate, "call_llm", lambda *a, **k: ("[]", None, None, usage))
    monkeypatch.setattr(generate, "save_registry", lambda d: None)

    _process_one_batch(
        client=None,
        batch=batch,
        rate_limiter=None,
        df=df,
        lock=threading.Lock(),
        save_reasoning=False,
        model="deepseek-v4-flash",
        thinking_enabled=False,
        reasoning_effort="high",
    )

    assert df.loc["1_001", "status"] == "awaiting_validation"
    assert df.loc["1_001", "call_cost_usd"] is None
    # Token counts are still tracked even though cost couldn't be priced.
    assert df.loc["1_001", "output_tokens"] == 50
