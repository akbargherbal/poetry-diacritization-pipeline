"""
Tests for pricing.py — the DeepSeek cost-calculation layer.

Core guarantee under test: pricing is entirely optional. A missing
pricing_config.py, an unpriced model, or missing token counts must all
degrade to `None`, never raise.
"""
import pytest

from poetry_diacritization import pricing

SAMPLE_PRICING = {
    "deepseek-v4-flash": {
        "currency": "USD",
        "unit_size": 1_000_000,
        "input_cache_miss_price": 0.14,
        "input_cache_hit_price": 0.0028,
        "output_price": 0.28,
    },
    "deepseek-v4-pro": {
        "currency": "USD",
        "unit_size": 1_000_000,
        "input_cache_miss_price": 0.435,
        "input_cache_hit_price": 0.003625,
        "output_price": 0.87,
    },
}


@pytest.fixture(autouse=True)
def _reset_warned_state():
    """These module-level "warn once" sets must not leak state between tests."""
    pricing._warned_missing_models.clear()
    pricing._warned_missing_config = False
    yield
    pricing._warned_missing_models.clear()
    pricing._warned_missing_config = False


# ---------------------------------------------------------------------------
# calculate_cost — happy path
# ---------------------------------------------------------------------------


def test_calculate_cost_matches_hand_computed_value():
    # 1,000,000 cache-hit + 1,000,000 cache-miss + 1,000,000 output tokens
    # at deepseek-v4-flash rates == exactly one unit of each price.
    cost = pricing.calculate_cost(
        "deepseek-v4-flash",
        cache_hit_tokens=1_000_000,
        cache_miss_tokens=1_000_000,
        output_tokens=1_000_000,
        pricing=SAMPLE_PRICING,
    )
    assert cost == pytest.approx(0.0028 + 0.14 + 0.28)


def test_calculate_cost_scales_linearly_with_tokens():
    cost = pricing.calculate_cost(
        "deepseek-v4-pro",
        cache_hit_tokens=500_000,
        cache_miss_tokens=200_000,
        output_tokens=100_000,
        pricing=SAMPLE_PRICING,
    )
    expected = (500_000 / 1_000_000) * 0.003625 + (200_000 / 1_000_000) * 0.435 + (
        100_000 / 1_000_000
    ) * 0.87
    assert cost == pytest.approx(expected)


def test_calculate_cost_zero_tokens_is_zero_cost():
    cost = pricing.calculate_cost(
        "deepseek-v4-flash",
        cache_hit_tokens=0,
        cache_miss_tokens=0,
        output_tokens=0,
        pricing=SAMPLE_PRICING,
    )
    assert cost == 0.0


# ---------------------------------------------------------------------------
# calculate_cost — must degrade to None, never raise
# ---------------------------------------------------------------------------


def test_calculate_cost_returns_none_for_unpriced_model():
    cost = pricing.calculate_cost(
        "some-future-model",
        cache_hit_tokens=100,
        cache_miss_tokens=100,
        output_tokens=100,
        pricing=SAMPLE_PRICING,
    )
    assert cost is None


def test_calculate_cost_returns_none_when_pricing_dict_is_empty():
    cost = pricing.calculate_cost(
        "deepseek-v4-flash",
        cache_hit_tokens=100,
        cache_miss_tokens=100,
        output_tokens=100,
        pricing={},
    )
    assert cost is None


@pytest.mark.parametrize(
    "cache_hit,cache_miss,output",
    [
        (None, 100, 100),
        (100, None, 100),
        (100, 100, None),
        (None, None, None),
    ],
)
def test_calculate_cost_returns_none_when_any_token_count_missing(cache_hit, cache_miss, output):
    cost = pricing.calculate_cost(
        "deepseek-v4-flash",
        cache_hit_tokens=cache_hit,
        cache_miss_tokens=cache_miss,
        output_tokens=output,
        pricing=SAMPLE_PRICING,
    )
    assert cost is None


def test_calculate_cost_returns_none_for_malformed_pricing_entry():
    broken_pricing = {"deepseek-v4-flash": {"unit_size": 1_000_000}}  # missing price keys
    cost = pricing.calculate_cost(
        "deepseek-v4-flash",
        cache_hit_tokens=100,
        cache_miss_tokens=100,
        output_tokens=100,
        pricing=broken_pricing,
    )
    assert cost is None


def test_calculate_cost_never_raises_on_zero_unit_size():
    broken_pricing = {
        "deepseek-v4-flash": {
            "unit_size": 0,
            "input_cache_hit_price": 0.1,
            "input_cache_miss_price": 0.1,
            "output_price": 0.1,
        }
    }
    cost = pricing.calculate_cost(
        "deepseek-v4-flash",
        cache_hit_tokens=100,
        cache_miss_tokens=100,
        output_tokens=100,
        pricing=broken_pricing,
    )
    assert cost is None


# ---------------------------------------------------------------------------
# load_pricing_config — optional file, must never raise
# ---------------------------------------------------------------------------


def test_load_pricing_config_returns_the_real_shipped_config():
    # Sanity check against the actual pricing_config.py shipped in the repo.
    loaded = pricing.load_pricing_config()
    assert "deepseek-v4-flash" in loaded
    assert "deepseek-v4-pro" in loaded
    assert loaded["deepseek-v4-flash"]["unit_size"] == 1_000_000


def test_load_pricing_config_returns_empty_dict_when_module_missing(monkeypatch):
    # Setting sys.modules[name] = None makes a *fresh* `import name` raise
    # ImportError — the same failure Python gives for a genuinely
    # deleted/renamed pricing_config.py. Also strip the cached attribute
    # off the parent package, since it was already imported once earlier
    # in this test session and Python would otherwise just hand back that
    # cached reference instead of re-raising.
    import sys

    import poetry_diacritization as pkg

    had_attr = hasattr(pkg, "pricing_config")
    if had_attr:
        monkeypatch.delattr(pkg, "pricing_config")
    monkeypatch.setitem(sys.modules, "poetry_diacritization.pricing_config", None)

    result = pricing.load_pricing_config()
    assert result == {}


def test_load_pricing_config_returns_empty_dict_when_dict_missing_from_module(monkeypatch):
    import sys
    import types

    import poetry_diacritization as pkg

    fake_module = types.ModuleType("poetry_diacritization.pricing_config")
    # deliberately no DEEPSEEK_PRICING attribute

    if hasattr(pkg, "pricing_config"):
        monkeypatch.delattr(pkg, "pricing_config")
    monkeypatch.setitem(sys.modules, "poetry_diacritization.pricing_config", fake_module)

    result = pricing.load_pricing_config()
    assert result == {}
