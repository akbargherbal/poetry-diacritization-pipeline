import pytest

from poetry_diacritization import text_normalize
from poetry_diacritization.text_normalize import normalize, texts_match


def test_none_input_returns_empty_string():
    assert normalize(None) == ""


def test_strips_tashkeel_diacritics():
    diacritized = "الشَّمْسُ"
    bare = "الشمس"
    assert normalize(diacritized) == normalize(bare)


def test_strips_tatweel_kashida():
    with_tatweel = "الـــشمس"
    without = "الشمس"
    assert normalize(with_tatweel) == normalize(without)


@pytest.mark.parametrize(
    "with_hamza_alef, plain_alef",
    [
        ("أحمد", "احمد"),
        ("إحسان", "احسان"),
        ("آمال", "امال"),
    ],
)
def test_alef_hamza_forms_normalize_to_bare_alef(with_hamza_alef, plain_alef):
    # normalize_alef must run before normalize_hamza (per the module's own
    # docstring) or hamza-bearing alefs collapse to bare hamza (ء) instead
    # of plain alef (ا). This test locks in the correct end result.
    assert normalize(with_hamza_alef) == normalize(plain_alef)
    assert "ء" not in normalize(with_hamza_alef)


def test_teh_marbuta_normalized_when_flag_on(monkeypatch):
    monkeypatch.setattr(text_normalize, "NORMALIZE_TEH", True)
    assert normalize("مدرسة") == normalize("مدرسه")


def test_teh_marbuta_not_normalized_when_flag_off(monkeypatch):
    monkeypatch.setattr(text_normalize, "NORMALIZE_TEH", False)
    assert normalize("مدرسة") != normalize("مدرسه")


def test_whitespace_collapses():
    assert normalize("كلمة    أخرى") == normalize("كلمة أخرى")
    # NORMALIZE_TEH is on by default, so compare against normalize() of the
    # trimmed string rather than the trimmed string verbatim.
    assert normalize("  محاطة بمسافات  ") == normalize("محاطة بمسافات")


def test_texts_match_delegates_to_normalize():
    assert texts_match("الشَّمْسُ", "الشمس") is True
    assert texts_match("كلمة", "كلمتان") is False
