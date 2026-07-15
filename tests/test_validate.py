import json
import os

import pytest

from poetry_diacritization import config, validate
from poetry_diacritization.validate import _parse_llm_json, _score_with_pyarud, run_validation_pass


# ---------------------------------------------------------------------------
# _parse_llm_json
# ---------------------------------------------------------------------------


def test_parse_plain_json_array():
    raw = json.dumps([{"id": "1_001", "sadr": "a", "ajuz": "b"}])
    assert _parse_llm_json(raw) == {"1_001": {"sadr": "a", "ajuz": "b"}}


def test_parse_json_wrapped_in_markdown_fence():
    raw = "```json\n" + json.dumps([{"id": "1_001", "sadr": "a", "ajuz": "b"}]) + "\n```"
    assert _parse_llm_json(raw) == {"1_001": {"sadr": "a", "ajuz": "b"}}


def test_parse_json_object_with_one_list_valued_key():
    raw = json.dumps({"verses": [{"id": "1_001", "sadr": "a", "ajuz": "b"}]})
    assert _parse_llm_json(raw) == {"1_001": {"sadr": "a", "ajuz": "b"}}


def test_parse_json_object_with_no_list_valued_key_raises():
    raw = json.dumps({"note": "no verses here"})
    with pytest.raises(ValueError):
        _parse_llm_json(raw)


def test_parse_invalid_json_raises_value_error():
    with pytest.raises(ValueError, match="invalid JSON"):
        _parse_llm_json("not json at all {{{")


def test_parse_empty_string_raises():
    with pytest.raises(ValueError, match="empty response"):
        _parse_llm_json("")
    with pytest.raises(ValueError, match="empty response"):
        _parse_llm_json("   ")


def test_parse_skips_items_missing_id():
    raw = json.dumps(
        [
            {"id": "1_001", "sadr": "a", "ajuz": "b"},
            {"sadr": "no id here", "ajuz": "x"},
        ]
    )
    result = _parse_llm_json(raw)
    assert list(result.keys()) == ["1_001"]


def test_parse_prefers_id_over_verse_id_when_both_present():
    raw = json.dumps([{"id": "from_id", "verse_id": "from_verse_id", "sadr": "a", "ajuz": "b"}])
    result = _parse_llm_json(raw)
    assert list(result.keys()) == ["from_id"]


def test_parse_not_a_list_or_dict_raises():
    with pytest.raises(ValueError, match="expected a JSON array"):
        _parse_llm_json(json.dumps("just a string"))


# ---------------------------------------------------------------------------
# _score_with_pyarud
# ---------------------------------------------------------------------------


class _FakeProcessor:
    def __init__(self, result=None, raise_exc=None):
        self._result = result
        self._raise_exc = raise_exc

    def process_poem(self, verses, meter_name):
        if self._raise_exc:
            raise self._raise_exc
        return self._result


def test_score_with_pyarud_success(monkeypatch):
    fake_result = {
        "verses": [
            {
                "score": 0.95,
                "input_pattern": "XYZ",
                "best_ref_pattern": "XYZ",
                "extra_verbose_field": "should not leak into detail",
            }
        ]
    }
    monkeypatch.setattr(validate, "_processor", _FakeProcessor(result=fake_result))

    score, detail, error = _score_with_pyarud("sadr", "ajuz", "wafer")

    assert error is None
    assert score == 0.95
    assert detail == {"input_pattern": "XYZ", "best_ref_pattern": "XYZ"}
    assert "extra_verbose_field" not in detail


def test_score_with_pyarud_error_key_in_result(monkeypatch):
    fake_result = {"verses": [{"error": "meter mismatch"}]}
    monkeypatch.setattr(validate, "_processor", _FakeProcessor(result=fake_result))

    score, detail, error = _score_with_pyarud("sadr", "ajuz", "wafer")

    assert score is None
    assert detail is None
    assert error == "meter mismatch"


def test_score_with_pyarud_exception_is_caught(monkeypatch):
    monkeypatch.setattr(validate, "_processor", _FakeProcessor(raise_exc=RuntimeError("crash")))

    score, detail, error = _score_with_pyarud("sadr", "ajuz", "wafer")

    assert score is None
    assert detail is None
    assert "crash" in error


# ---------------------------------------------------------------------------
# run_validation_pass
# ---------------------------------------------------------------------------


def _write_raw_response(call_id, verses):
    path = os.path.join(config.RAW_RESPONSES_DIR, f"{call_id}.txt")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(verses, f, ensure_ascii=False)
    return path


def test_run_validation_pass_noop_when_nothing_pending(make_registry_df):
    df = make_registry_df([{"verse_id": "1_001", "status": "passed"}])
    result = run_validation_pass(df)
    assert result.loc["1_001", "status"] == "passed"


def test_run_validation_pass_missing_raw_file_marks_failed_parse(make_registry_df):
    df = make_registry_df(
        [
            {
                "verse_id": "1_001",
                "status": "awaiting_validation",
                "last_call_id": "missing_call",
            }
        ]
    )
    result = run_validation_pass(df)
    assert result.loc["1_001", "status"] == "failed_parse"


def test_run_validation_pass_verse_missing_from_response_marks_failed_parse(make_registry_df):
    df = make_registry_df(
        [
            {"verse_id": "1_001", "status": "awaiting_validation", "last_call_id": "call1"},
            {"verse_id": "1_002", "status": "awaiting_validation", "last_call_id": "call1"},
        ]
    )
    # Only 1_001 is present in the raw response; 1_002 is missing entirely.
    _write_raw_response("call1", [{"id": "1_001", "sadr": df.loc["1_001", "sadr_raw"], "ajuz": df.loc["1_001", "ajuz_raw"]}])

    result = run_validation_pass(df)
    assert result.loc["1_002", "status"] == "failed_parse"


def test_run_validation_pass_text_mismatch_marks_failed_text_mismatch(make_registry_df):
    df = make_registry_df(
        [{"verse_id": "1_001", "status": "awaiting_validation", "last_call_id": "call1",
          "sadr_raw": "كلمة", "ajuz_raw": "أخرى"}]
    )
    # LLM changed the actual letters, not just diacritics.
    _write_raw_response("call1", [{"id": "1_001", "sadr": "كلمات مختلفة تماما", "ajuz": "أخرى"}])

    result = run_validation_pass(df)
    assert result.loc["1_001", "status"] == "failed_text_mismatch"


def test_run_validation_pass_invalid_json_marks_all_verses_failed_parse(make_registry_df):
    df = make_registry_df(
        [{"verse_id": "1_001", "status": "awaiting_validation", "last_call_id": "call1"}]
    )
    path = os.path.join(config.RAW_RESPONSES_DIR, "call1.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("not valid json {{{")

    result = run_validation_pass(df)
    assert result.loc["1_001", "status"] == "failed_parse"


def test_run_validation_pass_success_path_scores_verse(monkeypatch, make_registry_df):
    df = make_registry_df(
        [{"verse_id": "1_001", "status": "awaiting_validation", "last_call_id": "call1",
          "sadr_raw": "كلمة", "ajuz_raw": "أخرى", "meter": "wafer"}]
    )
    # Diacritized version of the same underlying letters -> fidelity passes.
    _write_raw_response("call1", [{"id": "1_001", "sadr": "كَلِمَة", "ajuz": "أُخْرَى"}])

    monkeypatch.setattr(
        validate, "_score_with_pyarud", lambda sadr, ajuz, meter: (0.97, {"input_pattern": "X"}, None)
    )

    result = run_validation_pass(df)

    assert result.loc["1_001", "status"] == "scored"
    assert result.loc["1_001", "pyarud_score"] == 0.97
    assert result.loc["1_001", "sadr_current"] == "كَلِمَة"


def test_run_validation_pass_pyarud_error_marks_failed_parse_with_detail(monkeypatch, make_registry_df):
    df = make_registry_df(
        [{"verse_id": "1_001", "status": "awaiting_validation", "last_call_id": "call1",
          "sadr_raw": "كلمة", "ajuz_raw": "أخرى"}]
    )
    _write_raw_response("call1", [{"id": "1_001", "sadr": "كَلِمَة", "ajuz": "أُخْرَى"}])

    monkeypatch.setattr(
        validate, "_score_with_pyarud", lambda sadr, ajuz, meter: (None, None, "pyarud blew up")
    )

    result = run_validation_pass(df)

    # NOTE: this is genuinely a bit misleading in the source — a pyarud
    # scoring failure is reported with the same status
    # ("failed_parse") as a JSON-parsing failure, even though nothing was
    # wrong with parsing here. Tested as specified; flagged in
    # docs/TESTING_STRATEGY.md as a candidate naming fix, not silently
    # patched around here.
    assert result.loc["1_001", "status"] == "failed_parse"
    assert result.loc["1_001", "pyarud_detail"] == {"error": "pyarud blew up"}


def test_run_validation_pass_shares_one_call_id_across_verses_parses_once(monkeypatch, make_registry_df):
    df = make_registry_df(
        [
            {"verse_id": "1_001", "status": "awaiting_validation", "last_call_id": "call1",
             "sadr_raw": "أ", "ajuz_raw": "ب"},
            {"verse_id": "1_002", "status": "awaiting_validation", "last_call_id": "call1",
             "sadr_raw": "ج", "ajuz_raw": "د"},
        ]
    )
    _write_raw_response(
        "call1",
        [
            {"id": "1_001", "sadr": "أ", "ajuz": "ب"},
            {"id": "1_002", "sadr": "ج", "ajuz": "د"},
        ],
    )
    monkeypatch.setattr(
        validate, "_score_with_pyarud", lambda sadr, ajuz, meter: (0.9, {}, None)
    )

    parse_calls = []
    real_parse = validate._parse_llm_json

    def counting_parse(raw_text):
        parse_calls.append(raw_text)
        return real_parse(raw_text)

    monkeypatch.setattr(validate, "_parse_llm_json", counting_parse)

    run_validation_pass(df)

    assert len(parse_calls) == 1  # parsed once for both verses, not twice
