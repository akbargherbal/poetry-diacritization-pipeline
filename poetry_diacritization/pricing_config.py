"""
DeepSeek batch-cost pricing — the ONLY file you need to touch to keep cost
tracking accurate as DeepSeek changes their rates.

This file is entirely optional. If you delete it, rename it, or leave a
model out of DEEPSEEK_PRICING, nothing breaks: `generate.py` will simply
leave the cost/token columns as None for whatever isn't priced. See
`pricing.py` for the (small) amount of code that reads this file.

Where to find current rates: https://api-docs.deepseek.com/quick_start/pricing

Each entry is keyed by the exact model string you'd pass to `--model` /
`config.DEFAULT_MODEL` (see `config.SUPPORTED_MODELS`). The fields below
mirror DeepSeek's own API `usage` object one-for-one on purpose — see
`pricing.py`'s `calculate_cost()` for exactly how each one is used.
"""

DEEPSEEK_PRICING = {
    "deepseek-v4-flash": {
        # ISO 4217 currency code all prices below are quoted in.
        "currency": "USD",
        # Prices are "per this many tokens" (DeepSeek quotes per 1M tokens).
        # If DeepSeek ever quotes "per 1K tokens" instead, change this to
        # 1_000 rather than rescaling every price by hand.
        "unit_size": 1_000_000,
        # Price per `unit_size` INPUT tokens that MISSED DeepSeek's context
        # cache (i.e. the normal/full input rate). Maps to the API
        # response's `usage.prompt_cache_miss_tokens`.
        "input_cache_miss_price": 0.14,
        # Price per `unit_size` INPUT tokens that HIT DeepSeek's context
        # cache (billed far cheaper — DeepSeek auto-caches repeated prompt
        # prefixes, no setup needed on your end). Maps to
        # `usage.prompt_cache_hit_tokens`.
        "input_cache_hit_price": 0.0028,
        # Price per `unit_size` OUTPUT tokens (the generated diacritized
        # JSON). Maps to `usage.completion_tokens`. Note: DeepSeek bills
        # "thinking"/reasoning tokens as output tokens too, so a batch run
        # with `--thinking` on will look more expensive here — that's
        # correct, not a bug.
        "output_price": 0.28,
    },
    "deepseek-v4-pro": {
        "currency": "USD",
        "unit_size": 1_000_000,
        "input_cache_miss_price": 0.435,
        "input_cache_hit_price": 0.003625,
        "output_price": 0.87,
    },
    # Add more models here as DeepSeek releases them, or if you point
    # `config.SUPPORTED_MODELS` at something else entirely. Any model
    # missing from this dict just won't get a cost computed for it — no
    # error, no crash.
}

# Fields intentionally NOT tracked here (and why), in case you're tempted
# to add them:
#   - Per-request minimums / rate-limit tiers: DeepSeek doesn't currently
#     have them; nothing in this pipeline needs them.
#   - Free-tier / promotional discount windows: these are time-boxed and
#     account-specific, not a property of the model itself — apply them
#     by editing the price fields above for the duration of the promo,
#     rather than adding a new field nothing else reads.
