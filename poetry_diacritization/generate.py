"""
Stage 1 — Generate.

Calls the LLM for every batch of verses that still needs an attempt.
Deliberately does NOT parse or validate the response — that's Stage 2
(validate.py), run as a separate step so you can re-validate without
burning API calls again. All this stage does:

  1. build batches from the registry (poem-grouped, <= BATCH_SIZE)
  2. call the LLM once per batch
  3. save the raw response text to disk
  4. stamp the registry rows involved: status -> 'awaiting_validation',
     pass_count += 1, last_model, last_call_id
  5. save the registry incrementally, after every single batch

If a batch's API call fails outright (network error, etc.), those verses
are left completely untouched — they keep whatever status they had, so
they're automatically picked up again on the next generate pass.
"""
import logging
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

from . import config
from .git_checkpoint import CheckpointCounter
from .llm_client import SlidingWindowRateLimiter, call_llm, setup_client
from .pricing import calculate_cost
from .registry import save_registry

log = logging.getLogger("poetry_diacritization")


def _save_raw_response(
    call_id: str, content: str, reasoning: str | None, save_reasoning: bool
) -> str:
    path = os.path.join(config.RAW_RESPONSES_DIR, f"{call_id}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(content or "")
    if reasoning and save_reasoning:
        rpath = os.path.join(config.RAW_RESPONSES_DIR, f"{call_id}_reasoning.txt")
        with open(rpath, "w", encoding="utf-8") as f:
            f.write(reasoning)
    return path


def _cost_fields_for_batch(model: str, usage: dict | None, n: int) -> dict | None:
    """
    Turn one call's token usage into a *per-verse* share of that call's
    tokens/cost — the call covers `n` verses (one poem-batch), so dividing
    by `n` here is what makes `df["cumulative_cost_usd"].sum()` an
    accurate registry-wide total later (see registry.cost_summary).

    Returns None (touch nothing) if the API gave us no usage block at all.
    Cost itself may still come back None inside the dict (e.g. no
    pricing_config.py, or `model` isn't priced) — that's fine, the token
    counts are still worth recording even when the cost can't be.
    """
    if usage is None or n <= 0:
        return None

    cost = calculate_cost(
        model,
        usage["cache_hit_tokens"],
        usage["cache_miss_tokens"],
        usage["completion_tokens"],
    )

    def _share(value):
        return value / n if value is not None else None

    return {
        "input_tokens_cache_hit": _share(usage["cache_hit_tokens"]),
        "input_tokens_cache_miss": _share(usage["cache_miss_tokens"]),
        "output_tokens": _share(usage["completion_tokens"]),
        "call_cost_usd": _share(cost),
    }


def _process_one_batch(
    client,
    batch,
    rate_limiter,
    df,
    lock,
    save_reasoning,
    model,
    thinking_enabled,
    reasoning_effort,
    provider=None,
    checkpoint: CheckpointCounter = None,
):
    verse_ids = [v["id"] for v in batch["verses"]]
    content, reasoning, error, usage = call_llm(
        client,
        batch["verses"],
        rate_limiter,
        model=model,
        thinking_enabled=thinking_enabled,
        reasoning_effort=reasoning_effort,
        provider=provider,
    )

    if error is not None or content is None:
        log.warning(f"Batch for poem {batch['poem_no']} ({verse_ids}) failed: {error}")
        return

    call_id = uuid.uuid4().hex[:12]
    _save_raw_response(call_id, content, reasoning, save_reasoning)
    cost_fields = _cost_fields_for_batch(model, usage, len(verse_ids))

    with lock:
        df.loc[verse_ids, "status"] = "awaiting_validation"
        df.loc[verse_ids, "last_call_id"] = call_id
        df.loc[verse_ids, "last_model"] = model
        df.loc[verse_ids, "pass_count"] = df.loc[verse_ids, "pass_count"] + 1
        if cost_fields is not None:
            df.loc[verse_ids, "input_tokens_cache_hit"] = cost_fields["input_tokens_cache_hit"]
            df.loc[verse_ids, "input_tokens_cache_miss"] = cost_fields["input_tokens_cache_miss"]
            df.loc[verse_ids, "output_tokens"] = cost_fields["output_tokens"]
            df.loc[verse_ids, "call_cost_usd"] = cost_fields["call_cost_usd"]
            if cost_fields["call_cost_usd"] is not None:
                df.loc[verse_ids, "cumulative_cost_usd"] = (
                    df.loc[verse_ids, "cumulative_cost_usd"].fillna(0)
                    + cost_fields["call_cost_usd"]
                )
        save_registry(df)

    log.info(f"Poem {batch['poem_no']}: got response for {len(verse_ids)} verses (call {call_id})")

    # Outside the registry lock -- a git push shouldn't hold up other
    # threads' registry writes.
    if checkpoint is not None:
        checkpoint.batch_done()


def run_generation_pass(
    df,
    client=None,
    model: str = None,
    thinking_enabled: bool = None,
    reasoning_effort: str = None,
    save_reasoning: bool = None,
    provider: str = None,
    checkpoint_every: int = None,
    checkpoint_enabled: bool = None,
):
    """Run one generation pass over everything in df that needs an LLM attempt.

    model/thinking_enabled/reasoning_effort default to config.py's DEFAULT_*
    values (resolved here, once, so every batch in the pass uses the same
    settings and `last_model` records what was actually used — not just
    whatever config.DEFAULT_MODEL happens to be at read time).

    save_reasoning defaults to config.SAVE_REASONING_ARTIFACTS (False) but can
    be overridden per call (that's how the CLI's --save-reasoning /
    --no-save-reasoning flags reach here).

    provider defaults to config.MODEL_PROVIDER ("deepseek" unless
    MODEL_PROVIDER=nvidia is set) and is used both for building the client
    (if one isn't passed in) and for shaping each LLM call's request body.

    checkpoint_every/checkpoint_enabled default to config's
    CHECKPOINT_EVERY_N_BATCHES/CHECKPOINT_ENABLED. Every `checkpoint_every`
    successfully completed batches, runtime/ is committed and pushed to
    the git remote (see git_checkpoint.py) — a no-op, non-fatal warning if
    that fails for any reason.
    """
    from .registry import make_batches
    import threading

    provider = provider or config.MODEL_PROVIDER
    if client is None:
        client = setup_client(provider=provider)
    if save_reasoning is None:
        save_reasoning = config.SAVE_REASONING_ARTIFACTS
    model = model or config.DEFAULT_MODEL
    if thinking_enabled is None:
        thinking_enabled = config.DEFAULT_THINKING_ENABLED
    reasoning_effort = reasoning_effort or config.DEFAULT_REASONING_EFFORT

    batches = make_batches(df)
    if not batches:
        log.info("Nothing pending — no batches to generate.")
        return df

    rate_limiter = SlidingWindowRateLimiter(max_requests=config.REQUESTS_PER_MINUTE)
    lock = threading.Lock()
    checkpoint = CheckpointCounter(every_n=checkpoint_every, enabled=checkpoint_enabled)

    start = time.time()
    with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as pool:
        futures = [
            pool.submit(
                _process_one_batch,
                client,
                batch,
                rate_limiter,
                df,
                lock,
                save_reasoning,
                model,
                thinking_enabled,
                reasoning_effort,
                provider,
                checkpoint,
            )
            for batch in batches
        ]
        for f in as_completed(futures):
            f.result()  # re-raise any unexpected exception loudly

    log.info(f"Generation pass done: {len(batches)} batches in {time.time() - start:.1f}s")
    return df
