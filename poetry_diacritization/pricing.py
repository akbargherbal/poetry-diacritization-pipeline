"""
Batch-cost calculation for DeepSeek API calls.

This module is the only thing that reads `pricing_config.py`, and it never
lets a pricing problem (missing file, missing model, malformed entry)
propagate up as an exception — cost tracking is a nice-to-have layered on
top of the pipeline, not a dependency of it. Worst case, cost/token columns
in the registry are left as None and everything else works exactly as it
did before this feature existed.
"""
import logging

log = logging.getLogger("poetry_diacritization")

# Log each "missing pricing" situation once per run, not once per batch.
_warned_missing_config = False
_warned_missing_models = set()


def load_pricing_config() -> dict:
    """Load DEEPSEEK_PRICING from pricing_config.py, if it exists.

    Returns {} — never raises — if pricing_config.py has been deleted,
    renamed, or doesn't define DEEPSEEK_PRICING.
    """
    global _warned_missing_config
    try:
        from . import pricing_config
    except ImportError:
        if not _warned_missing_config:
            log.warning(
                "pricing_config.py not found — cost tracking columns in the "
                "registry will be left as None. Add poetry_diacritization/"
                "pricing_config.py (see the shipped example) to enable it."
            )
            _warned_missing_config = True
        return {}

    return getattr(pricing_config, "DEEPSEEK_PRICING", {}) or {}


def calculate_cost(
    model: str,
    cache_hit_tokens,
    cache_miss_tokens,
    output_tokens,
    pricing: dict = None,
):
    """
    Cost (in the pricing config's currency, USD for DeepSeek) of one LLM
    call, given its token usage. Returns None — never raises — if:
      - `pricing_config.py` is missing, or has no entry for `model`
      - any of the token counts is None (e.g. the API response had no
        usage block, which the OpenAI-compatible SDK doesn't guarantee)
      - the pricing entry for `model` is malformed (missing a required key)

    `pricing` can be passed explicitly (mainly for tests); defaults to
    `load_pricing_config()`.
    """
    if pricing is None:
        pricing = load_pricing_config()

    rates = pricing.get(model)
    if rates is None:
        if model not in _warned_missing_models:
            log.warning(
                f"No DeepSeek pricing entry for model '{model}' — cost left as None "
                f"for this call. Add it to pricing_config.py to track spend on it."
            )
            _warned_missing_models.add(model)
        return None

    if cache_hit_tokens is None or cache_miss_tokens is None or output_tokens is None:
        return None

    try:
        unit_size = rates["unit_size"]
        cost = (
            (cache_hit_tokens / unit_size) * rates["input_cache_hit_price"]
            + (cache_miss_tokens / unit_size) * rates["input_cache_miss_price"]
            + (output_tokens / unit_size) * rates["output_price"]
        )
    except (KeyError, TypeError, ZeroDivisionError) as e:
        log.warning(f"Malformed pricing entry for model '{model}' ({e}) — cost left as None.")
        return None

    return round(cost, 8)
