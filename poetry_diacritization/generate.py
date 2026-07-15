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
from .llm_client import SlidingWindowRateLimiter, call_llm, setup_client
from .registry import save_registry

log = logging.getLogger("poetry_diacritization")


def _save_raw_response(call_id: str, content: str, reasoning: str | None) -> str:
    path = os.path.join(config.RAW_RESPONSES_DIR, f"{call_id}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(content or "")
    if reasoning:
        rpath = os.path.join(config.RAW_RESPONSES_DIR, f"{call_id}_reasoning.txt")
        with open(rpath, "w", encoding="utf-8") as f:
            f.write(reasoning)
    return path


def _process_one_batch(client, batch, rate_limiter, df, lock):
    verse_ids = [v["id"] for v in batch["verses"]]
    content, reasoning, error = call_llm(client, batch["verses"], rate_limiter)

    if error is not None or content is None:
        log.warning(f"Batch for poem {batch['poem_no']} ({verse_ids}) failed: {error}")
        return

    call_id = uuid.uuid4().hex[:12]
    _save_raw_response(call_id, content, reasoning)

    with lock:
        df.loc[verse_ids, "status"] = "awaiting_validation"
        df.loc[verse_ids, "last_call_id"] = call_id
        df.loc[verse_ids, "last_model"] = config.MODEL
        df.loc[verse_ids, "pass_count"] = df.loc[verse_ids, "pass_count"] + 1
        save_registry(df)

    log.info(f"Poem {batch['poem_no']}: got response for {len(verse_ids)} verses (call {call_id})")


def run_generation_pass(df, client=None):
    """Run one generation pass over everything in df that needs an LLM attempt."""
    from .registry import make_batches
    import threading

    if client is None:
        client = setup_client()

    batches = make_batches(df)
    if not batches:
        log.info("Nothing pending — no batches to generate.")
        return df

    rate_limiter = SlidingWindowRateLimiter(max_requests=config.REQUESTS_PER_MINUTE)
    lock = threading.Lock()

    start = time.time()
    with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as pool:
        futures = [
            pool.submit(_process_one_batch, client, batch, rate_limiter, df, lock)
            for batch in batches
        ]
        for f in as_completed(futures):
            f.result()  # re-raise any unexpected exception loudly

    log.info(f"Generation pass done: {len(batches)} batches in {time.time() - start:.1f}s")
    return df
