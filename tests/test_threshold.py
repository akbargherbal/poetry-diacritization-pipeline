from poetry_diacritization.threshold import apply_threshold, score_distribution


def test_apply_threshold_only_touches_scored_rows_by_default(make_registry_df):
    df = make_registry_df(
        [
            {"verse_id": "1_001", "status": "scored", "pyarud_score": 0.95},
            {"verse_id": "1_002", "status": "scored", "pyarud_score": 0.5},
            # Already resolved in an earlier pass — must be left alone.
            {"verse_id": "1_003", "status": "passed", "pyarud_score": 0.10},
        ]
    )
    apply_threshold(df, cutoff=0.90)

    assert df.loc["1_001", "status"] == "passed"
    assert df.loc["1_002", "status"] == "failed_prosody"
    assert df.loc["1_003", "status"] == "passed"  # untouched, not re-flipped


def test_apply_threshold_boundary_is_inclusive(make_registry_df):
    df = make_registry_df([{"verse_id": "1_001", "status": "scored", "pyarud_score": 0.90}])
    apply_threshold(df, cutoff=0.90)
    assert df.loc["1_001", "status"] == "passed"


def test_apply_threshold_include_rescored_reconsiders_everything(make_registry_df):
    df = make_registry_df(
        [
            {"verse_id": "1_001", "status": "passed", "pyarud_score": 0.92},
        ]
    )
    # Stricter cutoff should flip a previously-passed row back.
    apply_threshold(df, cutoff=0.95, include_rescored=True)
    assert df.loc["1_001", "status"] == "failed_prosody"


def test_apply_threshold_ignores_rows_without_a_score(make_registry_df):
    df = make_registry_df(
        [{"verse_id": "1_001", "status": "scored", "pyarud_score": None}]
    )
    apply_threshold(df, cutoff=0.90)
    # NaN score: neither >= nor < cutoff evaluates True, so status is left as-is.
    assert df.loc["1_001", "status"] == "scored"


def test_score_distribution_none_when_nothing_scored(make_registry_df):
    df = make_registry_df([{"verse_id": "1_001", "status": "pending", "pyarud_score": None}])
    assert score_distribution(df) is None


def test_score_distribution_returns_describe_output(make_registry_df):
    df = make_registry_df(
        [
            {"verse_id": "1_001", "pyarud_score": 0.9},
            {"verse_id": "1_002", "pyarud_score": 0.8},
        ]
    )
    dist = score_distribution(df)
    assert dist is not None
    assert dist["count"] == 2
