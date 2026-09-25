"""
Text preprocessing and token extraction utilities for blocking and candidate generation.
Re-exports and adapts normalization routines from src.preprocessing.
"""

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
]
