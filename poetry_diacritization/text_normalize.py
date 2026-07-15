"""
One normalize() function. Applied identically to the original undiacritized
text and to the (stripped-of-diacritics) LLM output. If the two normalized
strings are not byte-for-byte identical, the LLM changed something beyond
diacritics — reject the verse, don't run pyarud on it.

Order matters:
- normalize_alef_maksura must run BEFORE normalize_alef. Alef maqsura (ى)
  is etymologically a "shortened yeh" (that's literally what the Arabic
  name means) and is very commonly used interchangeably with regular yeh
  (ي) in word-final position — e.g. "الذى/الذي", "يشتهى/يشتهي". But
  araby.normalize_alef() collapses ى straight to plain alef (ا), which
  would erase that distinction and make it indistinguishable from a
  genuinely different word ending in ا. So we redirect ى -> ي first,
  while it's still unambiguous.
- normalize_alef must run before normalize_hamza, otherwise hamza-bearing
  alefs (e.g. أ) get collapsed to bare hamza (ء) instead of plain alef
  (ا), which would break the أ/ا equivalence this whole function exists
  to guarantee.
"""
import pyarabic.araby as araby

# Toggles here if you decide one of these equivalences should NOT be
# treated as such after seeing real failure data. One line each to flip.
NORMALIZE_TEH = True
NORMALIZE_ALEF_MAKSURA = True


def normalize(text: str) -> str:
    if text is None:
        return ""
    text = araby.strip_tashkeel(text)
    text = araby.strip_tatweel(text)
    text = araby.normalize_ligature(text)
    if NORMALIZE_ALEF_MAKSURA:
        # Must happen before normalize_alef (see module docstring) or the
        # ى/ي distinction is lost before we get a chance to use it.
        text = text.replace(araby.ALEF_MAKSURA, araby.YEH)
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
