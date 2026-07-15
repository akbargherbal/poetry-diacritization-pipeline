import logging

from . import config

log = logging.getLogger("poetry_diacritization")

EXPORT_COLUMNS = [
    "verse_id",
    "poem_no",
    "meter",
    "sadr_current",
    "ajuz_current",
    "pyarud_score",
]


def export_passed(df):
    passed = df[df["status"] == "passed"][EXPORT_COLUMNS].rename(
        columns={"sadr_current": "sadr", "ajuz_current": "ajuz"}
    )
    passed.to_pickle(config.EXPORT_PATH, protocol=4)
    passed.to_csv(config.EXPORT_CSV_PATH, index=False, encoding="utf-8-sig")
    log.info(
        f"Exported {len(passed)} passed verses to {config.EXPORT_PATH} and {config.EXPORT_CSV_PATH}"
    )
    return passed
