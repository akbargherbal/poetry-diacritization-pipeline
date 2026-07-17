import os
import threading

import pandas as pd
import pytest

from poetry_diacritization import config, generate
from poetry_diacritization.generate import (
    _process_one_batch,
    _save_raw_response,
    run_generation_pass,
)


# ---------------------------------------------------------------------------
# _save_raw_response — the reasoning-artifact toggle
# ---------------------------------------------------------------------------


def test_save_raw_response_writes_content_only_when_save_reasoning_false():
    path = _save_raw_response("call1", "the content", "some reasoning", save_reasoning=False)

    assert os.path.exists(path)
    with open(path, encoding="utf-8") as f:
        assert f.read() == "the content"

    reasoning_path = os.path.join(config.RAW_RESPONSES_DIR, "call1_reasoning.txt")
    assert not os.path.exists(reasoning_path)


def test_save_raw_response_writes_reasoning_file_when_enabled():
    _save_raw_response("call2", "the content", "some reasoning", save_reasoning=True)

    reasoning_path = os.path.join(config.RAW_RESPONSES_DIR, "call2_reasoning.txt")
    assert os.path.exists(reasoning_path)
    with open(reasoning_path, encoding="utf-8") as f:
        assert f.read() == "some reasoning"


def test_save_raw_response_no_reasoning_file_when_reasoning_is_none():
    _save_raw_response("call3", "the content", None, save_reasoning=True)

    reasoning_path = os.path.join(config.RAW_RESPONSES_DIR, "call3_reasoning.txt")
    assert not os.path.exists(reasoning_path)


def test_save_raw_response_writes_empty_content_file_when_content_none():
    path = _save_raw_response("call4", None, None, save_reasoning=False)
    with open(path, encoding="utf-8") as f:
        assert f.read() == ""


# ---------------------------------------------------------------------------
# _process_one_batch
# ---------------------------------------------------------------------------


def _batch_for(verse_ids, poem_no=1):
    return {
        "poem_no": poem_no,
        "meter": "wafer",
        "verses": [{"id": vid, "sadr": "s", "ajuz": "a"} for vid in verse_ids],
    }


def test_process_one_batch_success_updates_registry(monkeypatch, make_registry_df):
    df = make_registry_df(
        [{"verse_id": "1_001", "poem_no": 1, "status": "pending", "pass_count": 0}]
    )
    batch = _batch_for(["1_001"])

    monkeypatch.setattr(generate, "call_llm", lambda *a, **k: ("[]", None, None, None))
    save_calls = []
    monkeypatch.setattr(generate, "save_registry", lambda d: save_calls.append(1))

    lock = threading.Lock()
    _process_one_batch(
        client=None,
        batch=batch,
        rate_limiter=None,
        df=df,
        lock=lock,
        save_reasoning=False,
        model="deepseek-v4-flash",
        thinking_enabled=False,
        reasoning_effort="high",
    )

    assert df.loc["1_001", "status"] == "awaiting_validation"
    assert df.loc["1_001", "pass_count"] == 1
    assert df.loc["1_001", "last_call_id"] is not None
    assert df.loc["1_001", "last_model"] == "deepseek-v4-flash"
    assert save_calls  # save_registry was invoked


def test_process_one_batch_failure_leaves_registry_untouched(monkeypatch, make_registry_df):
    df = make_registry_df(
        [{"verse_id": "1_001", "poem_no": 1, "status": "pending", "pass_count": 0}]
    )
    before = df.copy(deep=True)
    batch = _batch_for(["1_001"])

    monkeypatch.setattr(generate, "call_llm", lambda *a, **k: (None, None, "boom", None))
    save_calls = []
    monkeypatch.setattr(generate, "save_registry", lambda d: save_calls.append(1))

    lock = threading.Lock()
    _process_one_batch(
        client=None,
        batch=batch,
        rate_limiter=None,
        df=df,
        lock=lock,
        save_reasoning=False,
        model="deepseek-v4-flash",
        thinking_enabled=False,
        reasoning_effort="high",
    )

    pd.testing.assert_frame_equal(df, before)
    assert save_calls == []


# ---------------------------------------------------------------------------
# run_generation_pass
# ---------------------------------------------------------------------------


def test_run_generation_pass_no_batches_is_noop(monkeypatch, make_registry_df):
    df = make_registry_df([{"verse_id": "1_001", "poem_no": 1, "status": "passed"}])

    def _boom(*a, **k):
        raise AssertionError("client should never be constructed when there's nothing to do")

    monkeypatch.setattr(generate, "setup_client", _boom)

    result = run_generation_pass(df, client=object())
    assert result is df


def test_run_generation_pass_uses_config_default_when_save_reasoning_not_given(
    monkeypatch, make_registry_df
):
    df = make_registry_df([{"verse_id": "1_001", "poem_no": 1, "status": "pending"}])
    monkeypatch.setattr(config, "SAVE_REASONING_ARTIFACTS", False)
    monkeypatch.setattr(generate, "call_llm", lambda *a, **k: ("[]", "some reasoning", None, None))

    run_generation_pass(df, client=object())

    call_id = df.loc["1_001", "last_call_id"]
    reasoning_path = os.path.join(config.RAW_RESPONSES_DIR, f"{call_id}_reasoning.txt")
    assert not os.path.exists(reasoning_path)


def test_run_generation_pass_explicit_save_reasoning_overrides_config_default(
    monkeypatch, make_registry_df
):
    df = make_registry_df([{"verse_id": "1_001", "poem_no": 1, "status": "pending"}])
    # Config default says "don't save"...
    monkeypatch.setattr(config, "SAVE_REASONING_ARTIFACTS", False)
    monkeypatch.setattr(generate, "call_llm", lambda *a, **k: ("[]", "some reasoning", None, None))

    # ...but an explicit True passed to the function wins.
    run_generation_pass(df, client=object(), save_reasoning=True)

    call_id = df.loc["1_001", "last_call_id"]
    reasoning_path = os.path.join(config.RAW_RESPONSES_DIR, f"{call_id}_reasoning.txt")
    assert os.path.exists(reasoning_path)


def test_run_generation_pass_records_last_model_used(monkeypatch, make_registry_df):
    # Regression test: previously this referenced a nonexistent
    # `config.MODEL`, which raised AttributeError on every real run.
    df = make_registry_df([{"verse_id": "1_001", "poem_no": 1, "status": "pending"}])
    monkeypatch.setattr(generate, "call_llm", lambda *a, **k: ("[]", None, None, None))

    run_generation_pass(df, client=object(), model="deepseek-v4-pro")

    assert df.loc["1_001", "last_model"] == "deepseek-v4-pro"


def test_run_generation_pass_defaults_model_from_config(monkeypatch, make_registry_df):
    df = make_registry_df([{"verse_id": "1_001", "poem_no": 1, "status": "pending"}])
    monkeypatch.setattr(config, "DEFAULT_MODEL", "deepseek-v4-flash")
    monkeypatch.setattr(generate, "call_llm", lambda *a, **k: ("[]", None, None, None))

    run_generation_pass(df, client=object())

    assert df.loc["1_001", "last_model"] == "deepseek-v4-flash"


def test_run_generation_pass_passes_thinking_settings_to_call_llm(monkeypatch, make_registry_df):
    df = make_registry_df([{"verse_id": "1_001", "poem_no": 1, "status": "pending"}])
    seen_kwargs = {}

    def fake_call_llm(
        client, verses, rate_limiter, model=None, thinking_enabled=None, reasoning_effort=None, provider=None
    ):
        seen_kwargs.update(
            model=model, thinking_enabled=thinking_enabled, reasoning_effort=reasoning_effort
        )
        return "[]", None, None, None

    monkeypatch.setattr(generate, "call_llm", fake_call_llm)

    run_generation_pass(
        df, client=object(), model="deepseek-v4-pro", thinking_enabled=True, reasoning_effort="max"
    )

    assert seen_kwargs == {
        "model": "deepseek-v4-pro",
        "thinking_enabled": True,
        "reasoning_effort": "max",
    }
