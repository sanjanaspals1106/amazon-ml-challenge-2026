"""
Text preprocessing and token extraction utilities for blocking and candidate generation.
Re-exports and adapts normalization routines from src.preprocessing.

Blocking-only additions (do not touch src.preprocessing.normalizer, so P3's pair-feature
normalization is unaffected):
  - normalize_name_blocking / normalize_address_blocking: wrap the shared normalize_name /
    normalize_address with (a) native-script -> Latin transliteration for names, (b) Indian
    city/state alias folding for addresses, and (c) domain-suffix stripping for domain-style
    business names ("intellecttechnologies.com").
"""

import re

from src.preprocessing.normalizer import (
    ABBREVIATIONS,
    ABBREVIATIONS_NAME,
    ADDRESS_NULL_PATTERN,
    PUNCT_RE,
    COMBINING_DIACRITICS_RE,
    WHITESPACE_RE,
    DIGITS_RE,
    TextNormalizer,
    fold_latin_diacritics,
    normalize_basic,
    normalize_name,
    normalize_address,
    sort_tokens,
    extract_prefixes_4,
    extract_house_number,
)
from .transliteration import transliterate_to_latin

__all__ = [
    "ABBREVIATIONS",
    "ABBREVIATIONS_NAME",
    "ADDRESS_NULL_PATTERN",
    "PUNCT_RE",
    "COMBINING_DIACRITICS_RE",
    "WHITESPACE_RE",
    "DIGITS_RE",
    "TextNormalizer",
    "fold_latin_diacritics",
    "normalize_name",
    "normalize_address",
    "sort_tokens",
    "extract_prefixes_4",
    "extract_house_number",
    "normalize_name_blocking",
    "normalize_address_blocking",
]

# Indian city/state aliases seen colliding across S1 vs S2/S3 (old name vs current official
# name, full state name vs 2-letter code). Keys and values are already lowercase, single
# tokens or short phrases as they appear post basic normalization (punctuation -> space).
# Folded to a single canonical token so blocking's exact/sparse channels see them as identical.
_ADDRESS_ALIASES = {
    "madras": "chennai",
    "bombay": "mumbai",
    "calcutta": "kolkata",
    "bangalore": "bengaluru",
    "gurgaon": "gurugram",
    "cochin": "kochi",
    "trivandrum": "thiruvananthapuram",
    "tamil nadu": "tn",
    "telangana": "tg",
    "andhra pradesh": "ap",
    "west bengal": "wb",
    "uttar pradesh": "up",
    "madhya pradesh": "mp",
    "himachal pradesh": "hp",
    "delhi": "dl",
    "new delhi": "dl",
    "national capital territory": "dl",
}
# Sort multi-word keys first so "tamil nadu" is folded before a lone "tamil"/"nadu" token match.
_ADDRESS_ALIAS_RE = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in sorted(_ADDRESS_ALIASES, key=len, reverse=True)) + r")\b"
)

_DOMAIN_SUFFIX_RE = re.compile(r"\b(www|com|net|org|in|co|biz|info)\b")


def _fold_address_aliases(text: str) -> str:
    if not text:
        return text
    return _ADDRESS_ALIAS_RE.sub(lambda m: _ADDRESS_ALIASES[m.group(1)], text)


def _strip_domain_noise(text: str) -> str:
    """Drop generic domain-suffix tokens (already space-separated by basic normalization's
    punctuation stripping) from domain-style names like 'intellecttechnologies com'."""
    if not text:
        return text
    cleaned = _DOMAIN_SUFFIX_RE.sub(" ", text)
    return WHITESPACE_RE.sub(" ", cleaned).strip()


def normalize_name_blocking(text) -> str:
    """
    Blocking-specific name normalization: transliterate native-script names to a Latin
    approximation (feeds the character n-gram channel), then run the standard normalize_name,
    then strip domain-suffix noise tokens.
    """
    if not text:
        return ""
    text = transliterate_to_latin(str(text))
    normalized = normalize_name(text)
    return _strip_domain_noise(normalized)


def normalize_address_blocking(text) -> str:
    """Blocking-specific address normalization: standard normalize_address, then fold
    known Indian city/state aliases to a canonical token."""
    if not text:
        return ""
    normalized = normalize_address(text)
    return _fold_address_aliases(normalized)

