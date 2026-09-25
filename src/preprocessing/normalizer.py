"""
Text normalization utilities for blocking / candidate generation.

Rules (deterministic, no external resources):
- lowercase, '&' -> ' and '
- fold Latin diacritics only (NFKD, drop U+0300-U+036F); Indic vowel signs are untouched
- ASCII punctuation / control chars / common Unicode quotes & dashes -> space
- collapse whitespace
- expand abbreviations token-wise (names use a smaller, less ambiguous map than addresses)
- addresses: drop NULL / N/A placeholders

All public functions take and return a single ``str`` (so they can be used with ``Series.apply``).
"""

import re
import unicodedata
from typing import Dict, List

# ASCII punctuation/symbols, control chars, common Unicode quotes/dashes/ellipsis/guillemets/danda/middle dot
PUNCT_RE = re.compile(r"[\x00-\x1f\x7f!-/:-@\[-`{-~‐-―‘-‟…«»।॥·]")
COMBINING_DIACRITICS_RE = re.compile(r"[̀-ͯ]")
WHITESPACE_RE = re.compile(r"\s+")
DIGITS_RE = re.compile(r"\d+")
# After punctuation removal: 'NULL', '<NULL>' -> 'null'; 'N/A' -> 'n a'; 'NA' -> 'na'
ADDRESS_NULL_PATTERN = re.compile(r"\b(?:null|n a|na)\b")

# Address abbreviations (token -> expansion)
ABBREVIATIONS: Dict[str, str] = {
    "pvt": "private", "ltd": "limited", "corp": "corporation", "inc": "incorporated", "co": "company",
    "rd": "road", "st": "street", "ave": "avenue", "av": "avenue", "dr": "drive", "ln": "lane",
    "ct": "court", "cir": "circle", "blvd": "boulevard", "bd": "boulevard", "hwy": "highway",
    "pkwy": "parkway", "ter": "terrace", "pl": "place", "apt": "apartment", "ste": "suite",
    "fl": "floor", "nr": "near", "opp": "opposite", "bldg": "building", "cv": "cove", "trl": "trail",
    "sq": "square", "mkt": "market", "r": "rue", "n": "north", "s": "south", "e": "east", "w": "west",
}

# Name abbreviations: only legal-form style tokens and unambiguous words
# (avoids Dr / St / Pl / Ct / N / S / E / W etc. which are ambiguous inside names).
_NAME_EXCLUDE = {"r", "n", "s", "e", "w", "av", "bd", "dr", "st", "pl", "ct", "cv", "fl"}
ABBREVIATIONS_NAME: Dict[str, str] = {k: v for k, v in ABBREVIATIONS.items() if k not in _NAME_EXCLUDE}


def _compile_abbrev(mapping: Dict[str, str]) -> "re.Pattern":
    return re.compile(r"\b(" + "|".join(sorted(map(re.escape, mapping), key=len, reverse=True)) + r")\b")


_ABBREV_RE = _compile_abbrev(ABBREVIATIONS)
_ABBREV_NAME_RE = _compile_abbrev(ABBREVIATIONS_NAME)


def fold_latin_diacritics(text: str) -> str:
    """Strip Latin combining diacritics (é -> e) without touching Indic vowel signs."""
    if not text or text.isascii():
        return text
    return COMBINING_DIACRITICS_RE.sub("", unicodedata.normalize("NFKD", text))


def normalize_basic(text: str) -> str:
    """Lowercase, punctuation -> space, collapse whitespace, strip. Keeps accents and native scripts."""
    if not text:
        return ""
    return WHITESPACE_RE.sub(" ", PUNCT_RE.sub(" ", text.lower())).strip()


def _normalize_ext(text: str, abbrev_re: "re.Pattern", abbrev_map: Dict[str, str]) -> str:
    if not text:
        return ""
    t = fold_latin_diacritics(text.lower().replace("&", " and "))
    t = WHITESPACE_RE.sub(" ", PUNCT_RE.sub(" ", t)).strip()
    return abbrev_re.sub(lambda m: abbrev_map[m.group(1)], t)


def normalize_name(text: str) -> str:
    """Normalize a business name (basic + '&'->'and' + diacritic fold + legal-form abbreviation expansion)."""
    return _normalize_ext(text, _ABBREV_NAME_RE, ABBREVIATIONS_NAME)


def normalize_address(text: str) -> str:
    """Normalize an address (as names, with address abbreviations and NULL / N/A placeholders removed)."""
    t = _normalize_ext(text, _ABBREV_RE, ABBREVIATIONS)
    if not t:
        return ""
    return WHITESPACE_RE.sub(" ", ADDRESS_NULL_PATTERN.sub(" ", t)).strip()


def sort_tokens(text: str) -> str:
    """Sort whitespace tokens alphabetically (order-invariant key)."""
    if not text:
        return ""
    return " ".join(sorted(text.split()))


def extract_prefixes_4(text: str) -> str:
    """4-character prefixes of every token with >= 3 chars (typo/suffix tolerant blocking tokens)."""
    if not text:
        return ""
    return " ".join(t[:4] for t in text.split() if len(t) >= 3)


def extract_house_number(text: str) -> str:
    """First numeric run in the text with leading zeros stripped ('' if none)."""
    if not text:
        return ""
    m = DIGITS_RE.search(text)
    return m.group(0).lstrip("0") or "0" if m else ""


class TextNormalizer:
    """Thin object wrapper around the module-level normalization functions."""

    def __init__(self) -> None:
        self.abbreviations = ABBREVIATIONS
        self.abbreviations_name = ABBREVIATIONS_NAME

    def normalize_basic(self, text: str) -> str:
        return normalize_basic(text)

    def normalize_name(self, text: str) -> str:
        return normalize_name(text)

    def normalize_address(self, text: str) -> str:
        return normalize_address(text)

    def sort_tokens(self, text: str) -> str:
        return sort_tokens(text)

    def extract_prefixes_4(self, text: str) -> str:
        return extract_prefixes_4(text)

    def extract_house_number(self, text: str) -> str:
        return extract_house_number(text)

    def fold_latin_diacritics(self, text: str) -> str:
        return fold_latin_diacritics(text)
