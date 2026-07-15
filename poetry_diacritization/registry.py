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
]

_save_lock = threading.Lock()


def build_registry_from_input(input_pickle_path: str = config.INPUT_PICKLE) -> pd.DataFrame:
    """Explode the batch-level input pickle into a flat, verse-level registry."""
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


def load_or_build_registry() -> pd.DataFrame:
    if os.path.exists(config.REGISTRY_PATH):
        df = pd.read_pickle(config.REGISTRY_PATH)
        log.info(f"Loaded existing registry with {len(df)} verses from {config.REGISTRY_PATH}")
        return df
    df = build_registry_from_input()
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
