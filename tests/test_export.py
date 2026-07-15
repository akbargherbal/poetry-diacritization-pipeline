import os

import pandas as pd

from poetry_diacritization import config
from poetry_diacritization.export import export_passed


def test_export_passed_only_includes_passed_rows(make_registry_df):
    df = make_registry_df(
        [
            {"verse_id": "1_001", "status": "passed", "sadr_current": "a", "ajuz_current": "b",
             "pyarud_score": 0.95},
            {"verse_id": "1_002", "status": "failed_prosody", "sadr_current": "c",
             "ajuz_current": "d", "pyarud_score": 0.5},
        ]
    )
    result = export_passed(df)
    assert result["verse_id"].tolist() == ["1_001"]


def test_export_passed_renames_current_columns(make_registry_df):
    df = make_registry_df(
        [
            {"verse_id": "1_001", "status": "passed", "sadr_current": "sadr-val",
             "ajuz_current": "ajuz-val", "pyarud_score": 0.95},
        ]
    )
    result = export_passed(df)
    assert list(result.columns) == ["verse_id", "poem_no", "meter", "sadr", "ajuz", "pyarud_score"]
    assert result.iloc[0]["sadr"] == "sadr-val"
    assert result.iloc[0]["ajuz"] == "ajuz-val"


def test_export_passed_writes_pickle_and_csv(make_registry_df):
    df = make_registry_df(
        [
            {"verse_id": "1_001", "status": "passed", "sadr_current": "a", "ajuz_current": "b",
             "pyarud_score": 0.95},
        ]
    )
    export_passed(df)

    assert os.path.exists(config.EXPORT_PATH)
    assert os.path.exists(config.EXPORT_CSV_PATH)

    reloaded = pd.read_pickle(config.EXPORT_PATH)
    assert len(reloaded) == 1

    csv_df = pd.read_csv(config.EXPORT_CSV_PATH, encoding="utf-8-sig")
    assert len(csv_df) == 1


def test_export_passed_empty_set_still_writes_valid_files(make_registry_df):
    df = make_registry_df(
        [{"verse_id": "1_001", "status": "pending", "sadr_current": None, "ajuz_current": None}]
    )
    result = export_passed(df)

    assert len(result) == 0
    assert os.path.exists(config.EXPORT_PATH)
    assert os.path.exists(config.EXPORT_CSV_PATH)


def test_export_passed_preserves_arabic_text_through_csv_round_trip(make_registry_df):
    df = make_registry_df(
        [
            {"verse_id": "1_001", "status": "passed", "sadr_current": "كَلِمَة",
             "ajuz_current": "أُخْرَى", "pyarud_score": 0.95},
        ]
    )
    export_passed(df)
    csv_df = pd.read_csv(config.EXPORT_CSV_PATH, encoding="utf-8-sig")
    assert csv_df.iloc[0]["sadr"] == "كَلِمَة"
    assert csv_df.iloc[0]["ajuz"] == "أُخْرَى"
