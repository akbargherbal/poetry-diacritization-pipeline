import os

import pandas as pd
import pytest

from poetry_diacritization import config, registry


# ---------------------------------------------------------------------------
# build_registry_from_input
# ---------------------------------------------------------------------------


def test_build_registry_from_input_explodes_verses(fake_input_pickle):
    path = fake_input_pickle(
        [
            {
                "poem_no": 1,
                "meter": "wafer",
                "verses": [
                    {"verse_id": "1_001", "sadr": "سدر1", "ajuz": "عجز1"},
                    {"verse_id": "1_002", "sadr": "سدر2", "ajuz": "عجز2"},
                ],
            },
            {
                "poem_no": 2,
                "meter": "taweel",
                "verses": [
                    {"verse_id": "2_001", "sadr": "سدر3", "ajuz": "عجز3"},
                ],
            },
        ]
    )

    df = registry.build_registry_from_input(path)

    assert len(df) == 3
    assert list(df.columns) == registry.REGISTRY_COLUMNS
    assert set(df["status"]) == {"pending"}
    assert (df["pass_count"] == 0).all()
    assert df.loc["1_001", "meter"] == "wafer"
    assert df.loc["2_001", "meter"] == "taweel"
    # index should be the verse_id itself
    assert df.index.tolist() == ["1_001", "1_002", "2_001"]


def test_build_registry_from_input_carries_extra_input_columns(fake_input_pickle):
    """Any column beyond poem_no/meter/DATA on the input pickle (e.g.
    POET_NAME, POET_RANK, batch_no, BATCH_SIZE) must be broadcast onto
    every verse row exploded from that batch -- not silently dropped."""
    path = fake_input_pickle(
        [
            {
                "poem_no": 1,
                "meter": "wafer",
                "POET_NAME": "المتنبي",
                "POET_RANK": 1,
                "batch_no": 7,
                "BATCH_SIZE": 2,
                "verses": [
                    {"verse_id": "1_001", "sadr": "سدر1", "ajuz": "عجز1"},
                    {"verse_id": "1_002", "sadr": "سدر2", "ajuz": "عجز2"},
                ],
            },
            {
                "poem_no": 2,
                "meter": "taweel",
                "POET_NAME": "ابو تمام",
                "POET_RANK": 2,
                "batch_no": 3,
                "BATCH_SIZE": 1,
                "verses": [
                    {"verse_id": "2_001", "sadr": "سدر3", "ajuz": "عجز3"},
                ],
            },
        ]
    )

    df = registry.build_registry_from_input(path)

    for col in ("POET_NAME", "POET_RANK", "batch_no", "BATCH_SIZE"):
        assert col in df.columns

    assert df.loc["1_001", "POET_NAME"] == "المتنبي"
    assert df.loc["1_002", "POET_NAME"] == "المتنبي"
    assert df.loc["2_001", "POET_NAME"] == "ابو تمام"
    assert df.loc["1_001", "POET_RANK"] == 1
    assert df.loc["1_001", "batch_no"] == 7
    assert df.loc["1_001", "BATCH_SIZE"] == 2

    # Core registry columns must still all be present alongside the extras.
    for col in registry.REGISTRY_COLUMNS:
        assert col in df.columns


def test_build_registry_from_input_normalizes_text(fake_input_pickle):
    path = fake_input_pickle(
        [
            {
                "poem_no": 1,
                "meter": "wafer",
                "verses": [
                    {"verse_id": "1_001", "sadr": "الشَّمْسُ", "ajuz": "القمر"},
                ],
            }
        ]
    )
    df = registry.build_registry_from_input(path)
    from poetry_diacritization.text_normalize import normalize

    assert df.loc["1_001", "sadr_norm"] == normalize("الشَّمْسُ")


def test_build_registry_from_input_raises_on_duplicate_verse_id(fake_input_pickle):
    path = fake_input_pickle(
        [
            {
                "poem_no": 1,
                "meter": "wafer",
                "verses": [
                    {"verse_id": "1_001", "sadr": "a", "ajuz": "b"},
                ],
            },
            {
                "poem_no": 2,
                "meter": "taweel",
                "verses": [
                    # Same verse_id reused across a different poem_no.
                    {"verse_id": "1_001", "sadr": "c", "ajuz": "d"},
                ],
            },
        ]
    )
    with pytest.raises(ValueError, match="duplicate verse_id"):
        registry.build_registry_from_input(path)


def test_build_registry_from_input_auto_detects_renamed_pickle(
    monkeypatch, tmp_path, fake_input_pickle
):
    """Regression test: swapping data/SAMPLE_POEMS.pkl for a differently
    named pickle (e.g. data/SAMPLE_100_BATCHES.pkl) used to crash with
    FileNotFoundError because the old default was bound to
    config.INPUT_PICKLE at import time. It should now be auto-detected.
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)
    monkeypatch.setattr(config, "BASE_DIR", str(tmp_path))
    monkeypatch.setattr(config, "INPUT_PICKLE", str(data_dir / "SAMPLE_POEMS.pkl"))
    monkeypatch.delenv(config.INPUT_PICKLE_ENV_VAR, raising=False)

    # fake_input_pickle writes into tmp_path (its own tmp_path fixture),
    # so build it there and copy into our isolated data_dir under a
    # different name than the historical default.
    src_path = fake_input_pickle(
        [
            {
                "poem_no": 1,
                "meter": "wafer",
                "verses": [{"verse_id": "1_001", "sadr": "a", "ajuz": "b"}],
            }
        ]
    )
    renamed = data_dir / "SAMPLE_100_BATCHES.pkl"
    renamed.write_bytes(open(src_path, "rb").read())

    df = registry.build_registry_from_input()  # no path given -> auto-detect
    assert len(df) == 1


# ---------------------------------------------------------------------------
# load_or_build_registry
# ---------------------------------------------------------------------------


def test_load_or_build_registry_builds_when_missing(monkeypatch, make_registry_df):
    # NOTE: build_registry_from_input's `input_pickle_path` default is bound
    # to config.INPUT_PICKLE at import time (`def f(path=config.INPUT_PICKLE)`),
    # so monkeypatching config.INPUT_PICKLE at test time does NOT change what
    # this default resolves to — a real gotcha in the source, not a test
    # artifact. We mock build_registry_from_input itself instead, which is
    # what load_or_build_registry actually calls.
    built_df = make_registry_df([{"verse_id": "1_001", "status": "pending"}])
    monkeypatch.setattr(registry, "build_registry_from_input", lambda: built_df)
    assert not os.path.exists(config.REGISTRY_PATH)

    df = registry.load_or_build_registry()

    assert len(df) == 1
    assert os.path.exists(config.REGISTRY_PATH)


def test_load_or_build_registry_loads_existing_without_rebuilding(
    monkeypatch, fake_input_pickle, make_registry_df
):
    # Put a registry on disk directly, bypassing build_registry_from_input.
    df = make_registry_df([{"verse_id": "9_001", "status": "passed"}])
    registry.save_registry(df)

    build_calls = []
    monkeypatch.setattr(
        registry,
        "build_registry_from_input",
        lambda *a, **k: build_calls.append(1) or df,
    )

    loaded = registry.load_or_build_registry()

    assert build_calls == []  # never rebuilt
    assert loaded.loc["9_001", "status"] == "passed"


# ---------------------------------------------------------------------------
# save_registry
# ---------------------------------------------------------------------------


def test_save_registry_is_atomic_and_round_trips(make_registry_df):
    df = make_registry_df([{"verse_id": "1_001", "status": "pending"}])
    registry.save_registry(df)

    assert os.path.exists(config.REGISTRY_PATH)
    assert not os.path.exists(config.REGISTRY_PATH + ".tmp")

    reloaded = pd.read_pickle(config.REGISTRY_PATH)
    assert reloaded.loc["1_001", "status"] == "pending"


# ---------------------------------------------------------------------------
# make_batches
# ---------------------------------------------------------------------------


def test_make_batches_only_includes_pending_statuses(make_registry_df):
    df = make_registry_df(
        [
            {"verse_id": "1_001", "poem_no": 1, "status": "pending"},
            {"verse_id": "1_002", "poem_no": 1, "status": "passed"},
            {"verse_id": "1_003", "poem_no": 1, "status": "failed_parse"},
            {"verse_id": "1_004", "poem_no": 1, "status": "scored"},
        ]
    )
    batches = registry.make_batches(df, batch_size=10)
    verse_ids = {v["id"] for b in batches for v in b["verses"]}
    assert verse_ids == {"1_001", "1_003"}


def test_make_batches_keeps_poems_separate(make_registry_df):
    df = make_registry_df(
        [
            {"verse_id": "1_001", "poem_no": 1, "status": "pending"},
            {"verse_id": "2_001", "poem_no": 2, "status": "pending"},
        ]
    )
    # Even with room to spare in a batch, different poems never share one.
    batches = registry.make_batches(df, batch_size=10)
    assert len(batches) == 2
    poem_nos = {b["poem_no"] for b in batches}
    assert poem_nos == {1, 2}


def test_make_batches_splits_large_poem_into_multiple_batches(make_registry_df):
    rows = [
        {"verse_id": f"1_{i:03d}", "poem_no": 1, "status": "pending"} for i in range(5)
    ]
    df = make_registry_df(rows)
    batches = registry.make_batches(df, batch_size=2)
    assert len(batches) == 3  # 2 + 2 + 1
    assert [len(b["verses"]) for b in batches] == [2, 2, 1]


def test_make_batches_orders_by_numeric_verse_suffix_not_row_order(make_registry_df):
    # Deliberately out of order in the DataFrame.
    df = make_registry_df(
        [
            {"verse_id": "1_003", "poem_no": 1, "status": "pending"},
            {"verse_id": "1_001", "poem_no": 1, "status": "pending"},
            {"verse_id": "1_002", "poem_no": 1, "status": "pending"},
        ]
    )
    batches = registry.make_batches(df, batch_size=10)
    assert len(batches) == 1
    ordered_ids = [v["id"] for v in batches[0]["verses"]]
    assert ordered_ids == ["1_001", "1_002", "1_003"]


def test_make_batches_returns_empty_list_when_nothing_pending(make_registry_df):
    df = make_registry_df([{"verse_id": "1_001", "poem_no": 1, "status": "passed"}])
    assert registry.make_batches(df, batch_size=10) == []


def test_status_counts(make_registry_df):
    df = make_registry_df(
        [
            {"verse_id": "1_001", "status": "pending"},
            {"verse_id": "1_002", "status": "pending"},
            {"verse_id": "1_003", "status": "passed"},
        ]
    )
    counts = registry.status_counts(df)
    assert counts["pending"] == 2
    assert counts["passed"] == 1
