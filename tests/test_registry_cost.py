"""
Tests for the cost-tracking additions to registry.py:
  - _ensure_columns / load_or_build_registry backfilling cost columns onto
    an older registry.pkl saved before they existed (no regression for
    users upgrading with an existing runtime/registry.pkl on disk)
  - cost_summary(), the pandas-native expense-tracking helper
"""
import pandas as pd
import pytest

from poetry_diacritization import registry


# ---------------------------------------------------------------------------
# _ensure_columns / load_or_build_registry — backward compatibility
# ---------------------------------------------------------------------------


def test_ensure_columns_backfills_missing_cost_columns():
    # Simulates a registry.pkl saved by a version of the code before cost
    # tracking existed: none of the new columns are present at all.
    old_df = pd.DataFrame(
        [{"verse_id": "1_001", "poem_no": 1, "status": "passed"}]
    ).set_index("verse_id", drop=False)
    assert "cumulative_cost_usd" not in old_df.columns

    migrated = registry._ensure_columns(old_df)

    assert list(migrated.columns) == list(dict.fromkeys(list(old_df.columns) + registry.REGISTRY_COLUMNS))
    for col in registry.REGISTRY_COLUMNS:
        assert col in migrated.columns
    # cumulative_cost_usd defaults to 0.0 (safely summable), not NaN.
    assert migrated.loc["1_001", "cumulative_cost_usd"] == 0.0
    assert pd.isna(migrated.loc["1_001", "call_cost_usd"])


def test_ensure_columns_is_a_noop_on_a_current_registry(make_registry_df):
    df = make_registry_df([{"verse_id": "1_001", "status": "pending"}])
    before_cols = list(df.columns)

    migrated = registry._ensure_columns(df)

    assert list(migrated.columns) == before_cols


def test_load_or_build_registry_migrates_an_old_registry_on_disk(monkeypatch):
    # Write an "old" registry (pre-cost-tracking schema) directly to disk,
    # bypassing save_registry/build_registry_from_input entirely, then load
    # it through the real code path.
    from poetry_diacritization import config

    old_df = pd.DataFrame(
        [{"verse_id": "1_001", "poem_no": 1, "status": "passed", "pass_count": 2}]
    ).set_index("verse_id", drop=False)
    old_df.to_pickle(config.REGISTRY_PATH, protocol=4)

    loaded = registry.load_or_build_registry()

    assert loaded.loc["1_001", "status"] == "passed"  # old data preserved
    assert loaded.loc["1_001", "cumulative_cost_usd"] == 0.0  # new column backfilled
    assert "input_tokens_cache_hit" in loaded.columns


# ---------------------------------------------------------------------------
# cost_summary
# ---------------------------------------------------------------------------


def test_cost_summary_empty_registry_returns_empty_frame(make_registry_df):
    df = make_registry_df([{"verse_id": "1_001", "status": "pending"}])  # never generated
    summary = registry.cost_summary(df)
    assert summary.empty


def test_cost_summary_totals_by_model(make_registry_df):
    df = make_registry_df(
        [
            {
                "verse_id": "1_001",
                "status": "awaiting_validation",
                "last_model": "deepseek-v4-flash",
                "input_tokens_cache_hit": 10,
                "input_tokens_cache_miss": 90,
                "output_tokens": 50,
                "cumulative_cost_usd": 0.01,
            },
            {
                "verse_id": "1_002",
                "status": "awaiting_validation",
                "last_model": "deepseek-v4-flash",
                "input_tokens_cache_hit": 20,
                "input_tokens_cache_miss": 80,
                "output_tokens": 40,
                "cumulative_cost_usd": 0.02,
            },
            {
                "verse_id": "2_001",
                "status": "awaiting_validation",
                "last_model": "deepseek-v4-pro",
                "input_tokens_cache_hit": 5,
                "input_tokens_cache_miss": 5,
                "output_tokens": 5,
                "cumulative_cost_usd": 0.05,
            },
            # Never sent to the LLM yet — must be excluded entirely.
            {"verse_id": "3_001", "status": "pending"},
        ]
    )

    summary = registry.cost_summary(df)

    assert summary.loc["deepseek-v4-flash", "verses"] == 2
    assert summary.loc["deepseek-v4-flash", "total_cost_usd"] == pytest.approx(0.03)
    assert summary.loc["deepseek-v4-pro", "verses"] == 1
    assert summary.loc["deepseek-v4-pro", "total_cost_usd"] == pytest.approx(0.05)
    assert summary.loc["TOTAL", "verses"] == 3
    assert summary.loc["TOTAL", "total_cost_usd"] == pytest.approx(0.08)
    assert "3_001" not in summary.index  # sanity: excluded verse never leaked in


def test_cost_summary_columns_present_even_when_empty(make_registry_df):
    df = make_registry_df([{"verse_id": "1_001", "status": "pending"}])
    summary = registry.cost_summary(df)
    expected_cols = {
        "verses",
        "input_tokens_cache_hit",
        "input_tokens_cache_miss",
        "output_tokens",
        "total_cost_usd",
    }
    assert expected_cols.issubset(set(summary.columns))
