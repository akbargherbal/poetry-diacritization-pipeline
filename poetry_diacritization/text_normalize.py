"""
One normalize() function. Applied identically to the original undiacritized
text and to the (stripped-of-diacritics) LLM output. If the two normalized
strings are not byte-for-byte identical, the LLM changed something beyond
diacritics — reject the verse, don't run pyarud on it.

Order matters: normalize_alef must run before normalize_hamza, otherwise
hamza-bearing alefs (e.g. أ) get collapsed to bare hamza (ء) instead of
plain alef (ا), which would break the أ/ا equivalence this whole function
exists to guarantee.
"""
import pyarabic.araby as araby

# Toggle here if you decide teh-marbuta/heh should NOT be treated as
# equivalent after seeing real failure data. One line to flip.
NORMALIZE_TEH = True


def normalize(text: str) -> str:
    if text is None:
        return ""
    text = araby.strip_tashkeel(text)
    text = araby.strip_tatweel(text)
    text = araby.normalize_ligature(text)
    text = araby.normalize_alef(text)
    text = araby.normalize_hamza(text)
    if NORMALIZE_TEH:
        text = araby.normalize_teh(text)
    # Collapse whitespace differences (extra spaces, etc.) — never a
    # meaningful change for our purposes.
    text = " ".join(text.split())
    return text


def texts_match(a: str, b: str) -> bool:
    return normalize(a) == normalize(b)
