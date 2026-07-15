"""
The registry is the single source of truth: one row per verse. It is built
once from the input batch-level pickle (exploded from DATA lists), then
mutated in place across as many generate/validate/threshold passes as you
want to run — including with different models.

The original input pickle is never modified.
"""
import logging
import os
import threading

import pandas as pd

from . import config
from .text_normalize import normalize

log = logging.getLogger("poetry_diacritization")

REGISTRY_COLUMNS = [
    "verse_id",
    "poem_no",
    "meter",
    "sadr_raw",
    "ajuz_raw",
    "sadr_norm",
    "ajuz_norm",
    "sadr_current",
    "ajuz_current",
    "status",
    "pyarud_score",
    "pyarud_detail",
    "pass_count",
    "last_model",
    "last_call_id",
    # -- Cost tracking (see pricing.py / pricing_config.py) -----------------
    # A generate() call covers a whole poem-batch (<= BATCH_SIZE verses),
    # not a single verse — so every field below is that call's total
    # divided evenly across the verses it covered. That's deliberate: it
    # means `df["cumulative_cost_usd"].sum()` (or any of the token columns)
    # gives an accurate running total for the whole registry, with no
    # de-duplication by `last_call_id` needed. See `cost_summary()`.
    # Always None if pricing_config.py is missing/incomplete, or the API
    # response carried no usage block — cost tracking never blocks a run.
    "input_tokens_cache_hit",   # this verse's share of the last call's prompt_cache_hit_tokens
    "input_tokens_cache_miss",  # this verse's share of the last call's prompt_cache_miss_tokens
    "output_tokens",            # this verse's share of the last call's completion_tokens
    "call_cost_usd",            # this verse's share of the *last* call's cost
    "cumulative_cost_usd",      # running total of call_cost_usd across every attempt so far
]

_save_lock = threading.Lock()


def build_registry_from_input(input_pickle_path: str = None) -> pd.DataFrame:
    """Explode the batch-level input pickle into a flat, verse-level registry.

    `input_pickle_path`, if given, is used as-is. Otherwise it's resolved
    via config.resolve_input_pickle() at *call* time (not import time --
    binding straight to config.INPUT_PICKLE in the signature was the bug
    that made swapping in a differently-named pickle crash with a
    FileNotFoundError instead of picking it up).
    """
    input_pickle_path = config.resolve_input_pickle(input_pickle_path)
    if not os.path.isfile(input_pickle_path):
        raise FileNotFoundError(
            f"Input pickle not found: {input_pickle_path}\n"
            f"Put your batch-level DataFrame pickle in "
            f"{os.path.join(config.BASE_DIR, 'data')}/ (it will be "
            "auto-detected if it's the only .pkl there), or point at it "
            "explicitly with --input path/to/your.pkl, or set "
            f"{config.INPUT_PICKLE_ENV_VAR}=path/to/your.pkl."
        )
    src = pd.read_pickle(input_pickle_path)

    rows = []
    for _, batch_row in src.iterrows():
        poem_no = batch_row["poem_no"]
        meter = batch_row["meter"]
        for verse in batch_row["DATA"]:
            rows.append(
                {
                    "verse_id": verse["verse_id"],
                    "poem_no": poem_no,
                    "meter": meter,
                    "sadr_raw": verse["sadr"],
                    "ajuz_raw": verse["ajuz"],
                    "sadr_norm": normalize(verse["sadr"]),
                    "ajuz_norm": normalize(verse["ajuz"]),
                    "sadr_current": None,
                    "ajuz_current": None,
                    "status": "pending",
                    "pyarud_score": None,
                    "pyarud_detail": None,
                    "pass_count": 0,
                    "last_model": None,
                    "last_call_id": None,
                    "input_tokens_cache_hit": None,
                    "input_tokens_cache_miss": None,
                    "output_tokens": None,
                    "call_cost_usd": None,
                    "cumulative_cost_usd": 0.0,
                }
            )

    df = pd.DataFrame(rows, columns=REGISTRY_COLUMNS)

    if df["verse_id"].duplicated().any():
        dupes = df.loc[df["verse_id"].duplicated(), "verse_id"].tolist()
        raise ValueError(
            f"Input pickle has duplicate verse_id values, refusing to build a "
            f"registry with an ambiguous primary key: {dupes}"
        )

    df = df.set_index("verse_id", drop=False)
    log.info(f"Built fresh registry with {len(df)} verses from {input_pickle_path}")
    return df


def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Backfill any registry columns added by a newer version of the code
    (e.g. the cost-tracking columns) onto an older registry.pkl saved
    before they existed — so upgrading never requires rebuilding the
    registry from scratch. New columns default to None/NaN (0.0 for
    cumulative_cost_usd, so it's always safely summable).
    """
    for col in REGISTRY_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0 if col == "cumulative_cost_usd" else None
    return df


def load_or_build_registry(input_pickle_path: str = None) -> pd.DataFrame:
    if os.path.exists(config.REGISTRY_PATH):
        df = pd.read_pickle(config.REGISTRY_PATH)
        df = _ensure_columns(df)
        log.info(f"Loaded existing registry with {len(df)} verses from {config.REGISTRY_PATH}")
        return df
    df = (
        build_registry_from_input(input_pickle_path)
        if input_pickle_path is not None
        else build_registry_from_input()
    )
    save_registry(df)
    return df


def save_registry(df: pd.DataFrame) -> None:
    """Thread-safe, atomic-ish incremental save. Call this often — it's cheap."""
    with _save_lock:
        tmp_path = config.REGISTRY_PATH + ".tmp"
        df.to_pickle(tmp_path, protocol=4)
        os.replace(tmp_path, config.REGISTRY_PATH)


def status_counts(df: pd.DataFrame) -> pd.Series:
    return df["status"].value_counts(dropna=False)


def cost_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-model spend breakdown, straight from the registry pickle:

        >>> import pandas as pd
        >>> from poetry_diacritization.registry import cost_summary
        >>> df = pd.read_pickle("runtime/registry.pkl")
        >>> cost_summary(df)
                            verses  input_tokens_cache_hit  input_tokens_cache_miss  output_tokens  total_cost_usd
        deepseek-v4-flash      120                   340.0                   9800.0         6100.0        0.003842
        TOTAL                  120                   340.0                   9800.0         6100.0        0.003842

    Every cost/token column on the registry already holds this verse's
    *share* of whatever call last touched it (see REGISTRY_COLUMNS), so a
    plain `.sum()` per model is an accurate total — no de-duplication by
    `last_call_id` needed. Verses never sent to the LLM yet (no
    `last_model`) are excluded. A verse revisited across multiple passes
    (possibly under different models) contributes its cost to whichever
    model most recently touched it — fine for a spend overview, but not a
    per-pass audit trail.
    """
    spent = df[df["last_model"].notna()]
    columns = [
        "verses",
        "input_tokens_cache_hit",
        "input_tokens_cache_miss",
        "output_tokens",
        "total_cost_usd",
    ]
    if spent.empty:
        return pd.DataFrame(columns=columns)

    summary = spent.groupby("last_model").agg(
        verses=("verse_id", "count"),
        input_tokens_cache_hit=("input_tokens_cache_hit", "sum"),
        input_tokens_cache_miss=("input_tokens_cache_miss", "sum"),
        output_tokens=("output_tokens", "sum"),
        total_cost_usd=("cumulative_cost_usd", "sum"),
    )
    summary.loc["TOTAL"] = summary.sum(numeric_only=True)
    return summary


def make_batches(df: pd.DataFrame, batch_size: int = config.BATCH_SIZE):
    """
    Group verses that still need an LLM attempt into per-poem batches of at
    most `batch_size`. Verses from the same poem stay together and stay in
    their original order; a poem with only 3 straggler verses left just gets
    a batch of 3 — batches are never padded with verses from other poems.
    """
    pending = df[df["status"].isin(config.NEEDS_GENERATION_STATUSES)].copy()
    if pending.empty:
        return []

    # Order within a poem by the numeric suffix of verse_id (…_024 etc.)
    def _order_key(verse_id: str):
        try:
            return int(str(verse_id).split("_")[-1])
        except ValueError:
            return 0

    pending["_order"] = pending["verse_id"].map(_order_key)
    pending = pending.sort_values(["poem_no", "_order"])

    batches = []
    for poem_no, group in pending.groupby("poem_no", sort=False):
        verse_ids = group["verse_id"].tolist()
        meter = group["meter"].iloc[0]
        for i in range(0, len(verse_ids), batch_size):
            chunk_ids = verse_ids[i : i + batch_size]
            chunk = group[group["verse_id"].isin(chunk_ids)]
            verses = [
                {"id": r.verse_id, "sadr": r.sadr_raw, "ajuz": r.ajuz_raw}
                for r in chunk.itertuples()
            ]
            batches.append({"poem_no": poem_no, "meter": meter, "verses": verses})

    log.info(f"Built {len(batches)} batches covering {len(pending)} pending verses")
    return batches
