"""
Script detection + transliteration helper for the blocking pipeline (P2 / India recall fix).

Purpose: Source 2 / Source 3 sometimes carry business names in a native Indian script
(Devanagari, Tamil, Telugu, Kannada, Malayalam, Bengali, Gurmukhi, Gujarati, Oriya) while the
Source 1 query name is in Latin script (English). Since blocking is purely token/character
based, these pairs have zero surface overlap and are never retrieved, regardless of K.

This module:
  1. Detects the dominant Indic script of a string using Unicode code-point ranges only
     (no ML model, no external data -- just fixed block boundaries, so it satisfies the
     "no pretrained external models or extra data" constraint).
  2. Transliterates that script to a Latin/ASCII approximation using `indic_transliteration`
     (MIT-licensed, pure Python, rule-based Unicode transliteration SCHEME TABLES -- not a
     trained model of any kind).
  3. Leaves ASCII / already-Latin text untouched (fast path, no library call).

Transliteration is phonetic and approximate (e.g. Tamil often renders "B" as "P", so
"Bombay" <-> Tamil-transliterated "Pampay" will still not be byte-identical). It is intended
to feed the character n-gram channel (see candidate_generator.py), which tolerates this kind
of partial/fuzzy overlap far better than exact token matching does -- not to produce a perfect
transliteration.
"""

from functools import lru_cache
from typing import Optional

from indic_transliteration import sanscript

_SCRIPT_RANGES = [
    (0x0900, 0x097F, sanscript.DEVANAGARI),
    (0x0B80, 0x0BFF, sanscript.TAMIL),
    (0x0C00, 0x0C7F, sanscript.TELUGU),
    (0x0C80, 0x0CFF, sanscript.KANNADA),
    (0x0D00, 0x0D7F, sanscript.MALAYALAM),
    (0x0980, 0x09FF, sanscript.BENGALI),
    (0x0A00, 0x0A7F, sanscript.GURMUKHI),
    (0x0A80, 0x0AFF, sanscript.GUJARATI),
    (0x0B00, 0x0B7F, sanscript.ORIYA),
]


def detect_indic_script(text: str) -> Optional[str]:
    """
    Return the indic_transliteration scheme name for the dominant Indic script in `text`,
    or None if the string is plain ASCII / Latin / has no recognized Indic script majority.
    Pure Unicode code-point range lookup -- no model, no external data.
    """
    if not text or text.isascii():
        return None
    counts = {}
    for ch in text:
        cp = ord(ch)
        for lo, hi, name in _SCRIPT_RANGES:
            if lo <= cp <= hi:
                counts[name] = counts.get(name, 0) + 1
                break
    if not counts:
        return None
    return max(counts, key=counts.get)


@lru_cache(maxsize=200_000)
def transliterate_to_latin(text: str) -> str:
    """
    Transliterate a native-script business name/address to a Latin-script approximation.
    ASCII input is returned unchanged (fast path). Cached, since the same corpus string is
    normalized repeatedly across batches.
    """
    if not text or text.isascii():
        return text
    script = detect_indic_script(text)
    if script is None:
        return text
    try:
        return sanscript.transliterate(text, script, sanscript.ITRANS)
    except Exception:
        # Never let a transliteration edge case break the pipeline; fall back to original text.
        return text
