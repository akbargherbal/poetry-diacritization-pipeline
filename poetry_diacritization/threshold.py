"""
Stage 3 — Threshold.

pyarud gives a continuous 0.0-1.0 score, not a pass/fail. This stage is
the only place a cutoff is applied, and it's applied to the *stored*
score — never re-computed — so changing your mind about 0.90 vs 0.95
costs one re-run of this function, not another API pass.

Safe to call repeatedly / with a different cutoff each time: it only
touches rows currently in 'scored' state, or (if you pass
include_rescored=True) any row that has a pyarud_score at all, letting
you retroactively loosen or tighten the bar on the whole registry.
"""
import logging

from .registry import save_registry

log = logging.getLogger("poetry_diacritization")


def apply_threshold(df, cutoff: float = 0.90, include_rescored: bool = False):
    if include_rescored:
        target = df["pyarud_score"].notna()
    else:
        target = df["status"] == "scored"

    passed = target & (df["pyarud_score"] >= cutoff)
    failed = target & (df["pyarud_score"] < cutoff)

    df.loc[passed, "status"] = "passed"
    df.loc[failed, "status"] = "failed_prosody"

    log.info(
        f"Threshold {cutoff}: {passed.sum()} newly passed, {failed.sum()} marked failed_prosody"
    )
    save_registry(df)
    return df


def score_distribution(df):
    """Quick look at where your scores land, to help pick a cutoff."""
    scored = df[df["pyarud_score"].notna()]
    if scored.empty:
        return None
    return scored["pyarud_score"].describe()
