"""
Stage 2 — Validate.

Runs entirely offline against whatever Stage 1 already saved to disk — no
API calls happen here, so you can re-run this as many times as you like
(e.g. after tweaking the normalize() pipeline) without spending tokens.

For every verse currently 'awaiting_validation':
  1. parse the raw JSON response for its call_id (once per call_id, not
     once per verse — several verses share one raw file)
  2. salvage whichever verse_ids are present and well-formed; anything
     sent but missing from the response -> 'failed_parse'
  3. check text fidelity via normalize() — LLM must not have changed the
     actual letters, only added diacritics -> 'failed_text_mismatch'
  4. if fidelity passes, score with pyarud (wrapped in try/except so a
     pyarud crash marks only that verse, not the whole pass) -> 'scored'

'scored' is a neutral holding state: apply_threshold() (in threshold.py)
is what turns a score into 'passed' or 'failed_prosody', on demand, with
whatever cutoff you decide on after looking at the score distribution.
"""
import json
import logging
import os
import re

from pyarud.processor import ArudhProcessor

from . import config
from .registry import save_registry
from .text_normalize import normalize

log = logging.getLogger("poetry_diacritization")

_processor = ArudhProcessor()

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def _parse_llm_json(raw_text: str):
    """Returns a dict {verse_id: {'sadr': ..., 'ajuz': ...}} or raises ValueError."""
    if not raw_text or not raw_text.strip():
        raise ValueError("empty response")

    cleaned = _FENCE_RE.sub("", raw_text.strip())

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"invalid JSON: {e}")

    if isinstance(parsed, dict):
        # Some models wrap the array in a container key. Take the first
        # list-valued entry rather than guessing a specific key name.
        list_values = [v for v in parsed.values() if isinstance(v, list)]
        if not list_values:
            raise ValueError("JSON object contains no array of verses")
        parsed = list_values[0]

    if not isinstance(parsed, list):
        raise ValueError(f"expected a JSON array, got {type(parsed).__name__}")

    result = {}
    for item in parsed:
        if not isinstance(item, dict):
            continue
        vid = item.get("id") or item.get("verse_id")
        if vid is None:
            continue
        result[str(vid)] = {"sadr": item.get("sadr"), "ajuz": item.get("ajuz")}
    return result


def _score_with_pyarud(sadr: str, ajuz: str, meter: str):
    """Returns (score, detail_dict_or_None, error_or_None). Never raises."""
    try:
        result = _processor.process_poem(verses=[(sadr, ajuz)], meter_name=meter)
        verse_result = result.get("verses", [{}])[0]
        if "error" in verse_result:
            return None, None, verse_result["error"]
        score = verse_result.get("score")
        # Keep only a small, useful summary — not the full foot-by-foot
        # breakdown — to keep the registry pickle light.
        detail = {
            "input_pattern": verse_result.get("input_pattern"),
            "best_ref_pattern": verse_result.get("best_ref_pattern"),
        }
        return score, detail, None
    except Exception as e:
        return None, None, str(e)


def reset_for_revalidation(df, statuses=("failed_text_mismatch", "failed_parse")):
    """
    Send rows back through run_validation_pass without calling the LLM
    again — for when you've changed normalize() or the parsing logic and
    want to see whether previously-failed verses now pass, using the raw
    response already saved on disk.

    Only rows with a `last_call_id` are eligible: that's what points at
    the saved runtime/raw_responses/<call_id>.txt file run_validation_pass
    needs. A row with no last_call_id has never actually been generated
    (e.g. still 'pending') — resetting it here would be a no-op at best,
    so it's left alone; it'll get picked up by the next `generate` pass
    instead, same as always.

    `failed_prosody` is deliberately NOT in the default statuses: those
    verses already passed the fidelity check and got a real pyarud score
    — re-parsing the same raw response can't change that outcome, so
    that's threshold.py's job (its `include_rescored` flag), not this
    function's.

    Safe to call repeatedly / after any code change: it only ever moves
    rows *into* 'awaiting_validation', never invents new LLM output.
    """
    eligible = df["status"].isin(statuses) & df["last_call_id"].notna()
    skipped = df["status"].isin(statuses) & df["last_call_id"].isna()

    reset_ids = df.index[eligible].tolist()
    skipped_ids = df.index[skipped].tolist()

    df.loc[eligible, "status"] = "awaiting_validation"

    log.info(
        f"Reset {len(reset_ids)} verse(s) for revalidation from {list(statuses)}"
        + (f"; skipped {len(skipped_ids)} with no saved raw response" if skipped_ids else "")
    )
    save_registry(df)
    return df, reset_ids, skipped_ids


def run_validation_pass(df):
    pending = df[df["status"] == "awaiting_validation"]
    if pending.empty:
        log.info("Nothing awaiting validation.")
        return df

    call_ids = pending["last_call_id"].dropna().unique().tolist()
    log.info(f"Validating {len(pending)} verses across {len(call_ids)} raw responses")

    for call_id in call_ids:
        group = df[(df["status"] == "awaiting_validation") & (df["last_call_id"] == call_id)]
        verse_ids = group.index.tolist()

        raw_path = os.path.join(config.RAW_RESPONSES_DIR, f"{call_id}.txt")
        try:
            with open(raw_path, encoding="utf-8") as f:
                raw_text = f.read()
        except FileNotFoundError:
            log.error(f"Raw response file missing for call {call_id}: {raw_path}")
            df.loc[verse_ids, "status"] = "failed_parse"
            continue

        try:
            parsed_map = _parse_llm_json(raw_text)
        except ValueError as e:
            log.warning(f"Call {call_id}: {e}")
            df.loc[verse_ids, "status"] = "failed_parse"
            save_registry(df)
            continue

        for verse_id in verse_ids:
            row = df.loc[verse_id]
            candidate = parsed_map.get(str(verse_id))

            if candidate is None or not candidate.get("sadr") or not candidate.get("ajuz"):
                df.loc[verse_id, "status"] = "failed_parse"
                continue

            sadr_candidate = candidate["sadr"]
            ajuz_candidate = candidate["ajuz"]

            df.loc[verse_id, "sadr_current"] = sadr_candidate
            df.loc[verse_id, "ajuz_current"] = ajuz_candidate

            sadr_ok = normalize(sadr_candidate) == normalize(row["sadr_raw"])
            ajuz_ok = normalize(ajuz_candidate) == normalize(row["ajuz_raw"])

            if not (sadr_ok and ajuz_ok):
                df.loc[verse_id, "status"] = "failed_text_mismatch"
                continue

            score, detail, pyarud_error = _score_with_pyarud(
                sadr_candidate, ajuz_candidate, row["meter"]
            )
            if pyarud_error is not None:
                log.warning(f"pyarud error on {verse_id}: {pyarud_error}")
                df.loc[verse_id, "status"] = "failed_parse"
                df.at[verse_id, "pyarud_detail"] = {"error": pyarud_error}
                continue

            df.loc[verse_id, "status"] = "scored"
            df.loc[verse_id, "pyarud_score"] = score
            df.at[verse_id, "pyarud_detail"] = detail

        save_registry(df)
        log.info(f"Call {call_id}: validated {len(verse_ids)} verses")

    return df
