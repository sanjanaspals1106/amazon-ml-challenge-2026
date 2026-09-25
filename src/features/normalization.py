"""
Text normalization utilities for pair feature extraction (M2).
Reuses and refines patterns discovered during dataset reconnaissance.
"""

import re
import unicodedata
from typing import Dict, List, Optional, Set, Tuple
import pandas as pd
import numpy as np

# Punctuation including common Unicode dashes, quotation marks, and Indic punctuation
PUNCT_RE = re.compile(r"[!-/:-@\[-`{-~‐-―‘-‟…«»।॥·]")
COMB_RE = re.compile(r"[̀-ͯ]")
WHITESPACE_RE = re.compile(r"\s+")
INDIC_RE = re.compile(r"[\u0900-\u0D7F]")  # Devanagari, Bengali, Gurmukhi, Gujarati, Oriya, Tamil, Telugu, Kannada, Malayalam

# Common address and entity abbreviations
ABBR_ADDRESS: Dict[str, str] = {
    "pvt": "private", "ltd": "limited", "corp": "corporation", "inc": "incorporated", "co": "company",
    "rd": "road", "st": "street", "ave": "avenue", "av": "avenue", "dr": "drive", "ln": "lane",
    "ct": "court", "cir": "circle", "blvd": "boulevard", "bd": "boulevard", "hwy": "highway",
    "pkwy": "parkway", "ter": "terrace", "pl": "place", "apt": "apartment", "ste": "suite",
    "fl": "floor", "nr": "near", "opp": "opposite", "bldg": "building", "cv": "cove",
    "trl": "trail", "sq": "square", "mkt": "market", "r": "rue", "n": "north", "s": "south",
    "e": "east", "w": "west",
}

# Unambiguous corporate abbreviations for business names
ABBR_NAME: Dict[str, str] = {
    k: v for k, v in ABBR_ADDRESS.items()
    if k not in ("r", "n", "s", "e", "w", "av", "bd", "dr", "st", "pl", "ct", "cv", "fl")
}

# Legal form taxonomy for canonical comparison
LEGAL_FORMS_CANONICAL: Dict[str, str] = {
    "corporation": "corp", "corp": "corp",
    "incorporated": "inc", "inc": "inc",
    "limited": "ltd", "ltd": "ltd",
    "private limited": "pvt_ltd", "pvt ltd": "pvt_ltd",
    "private": "pvt", "pvt": "pvt",
    "company": "co", "co": "co",
    "llc": "llc", "l l c": "llc",
    "llp": "llp", "l l p": "llp",
    "gmbh": "gmbh",
    "sa": "sa", "s a": "sa",
    "sarl": "sarl", "s a r l": "sarl",
    "sas": "sas", "s a s": "sas",
    "plc": "plc",
}

# Compiled regex for fast abbreviation expansion
def _build_abbr_regex(abbr_dict: Dict[str, str]) -> re.Pattern:
    patterns = sorted(abbr_dict.keys(), key=len, reverse=True)
    return re.compile(r"\b(" + "|".join(re.escape(k) for k in patterns) + r")\b", re.IGNORECASE)

ABBR_NAME_RE = _build_abbr_regex(ABBR_NAME)
ABBR_ADDR_RE = _build_abbr_regex(ABBR_ADDRESS)

PLACEHOLDER_ADDR_RE = re.compile(r"\b(null|n a|na|none|nan|not available|unknown)\b", re.IGNORECASE)

# House number extraction regex
HOUSE_NUM_RE = re.compile(r"\b(\d+)\b")

# US States and abbreviations
US_STATES = [
    "alabama", "alaska", "arizona", "arkansas", "california", "colorado", "connecticut",
    "delaware", "florida", "georgia", "hawaii", "idaho", "illinois", "indiana", "iowa",
    "kansas", "kentucky", "louisiana", "maine", "maryland", "massachusetts", "michigan",
    "minnesota", "mississippi", "missouri", "montana", "nebraska", "nevada", "new hampshire",
    "new jersey", "new mexico", "new york", "north carolina", "north dakota", "ohio",
    "oklahoma", "oregon", "pennsylvania", "rhode island", "south carolina", "south dakota",
    "tennessee", "texas", "utah", "vermont", "virginia", "washington", "west virginia",
    "wisconsin", "wyoming", "district of columbia"
]
US_STATE_CODES = [
    "al", "ak", "az", "ar", "ca", "co", "ct", "de", "fl", "ga", "hi", "id", "il", "in",
    "ia", "ks", "ky", "la", "me", "md", "ma", "mi", "mn", "ms", "mo", "mt", "ne", "nv",
    "nh", "nj", "nm", "ny", "nc", "nd", "oh", "ok", "or", "pa", "ri", "sc", "sd", "tn",
    "tx", "ut", "vt", "va", "wa", "wv", "wi", "wy", "dc"
]
US_STATE_MAP = dict(zip(US_STATES, US_STATE_CODES))
for c in US_STATE_CODES:
    US_STATE_MAP[c] = c

US_STATE_RE = re.compile(
    r"(?i)(?:^|[\s,])(" + "|".join(re.escape(s) for s in US_STATES) + "|" +
    "|".join(re.escape(c) for c in US_STATE_CODES) + r")(?=$|[\s,])"
)

# Postal code patterns (US: 5 digit or 5+4, India: 6 digit, France: 5 digit)
POSTAL_US_RE = re.compile(r"\b\d{5}(?:-\d{4})?\b")
POSTAL_INDIA_RE = re.compile(r"(?<![\d/-])[1-9]\d{2}\s?\d{3}(?![\d/-])")
POSTAL_FRANCE_RE = re.compile(r"\b(0[1-9]|[1-8]\d|9[0-8])\d{3}\b")


def fold_diacritics_str(text: str) -> str:
    """Strip Latin combining diacritics only (NFKD then drop U+0300-036F); Indic untouched."""
    if not text or text.isascii():
        return text
    decomposed = unicodedata.normalize("NFKD", text)
    return COMB_RE.sub("", decomposed)


def norm_basic_str(text: str) -> str:
    """Lowercase, punctuation -> space, collapse whitespace, strip. Keeps accents & native scripts."""
    if not text or pd.isna(text):
        return ""
    text = str(text).lower()
    text = PUNCT_RE.sub(" ", text)
    text = WHITESPACE_RE.sub(" ", text)
    return text.strip()


def norm_ext_name_str(text: str) -> str:
    """Basic normalization + diacritic fold + '&' -> 'and' + name abbreviation expansion."""
    if not text or pd.isna(text):
        return ""
    text = str(text).lower().replace("&", " and ")
    text = fold_diacritics_str(text)
    text = PUNCT_RE.sub(" ", text)
    text = WHITESPACE_RE.sub(" ", text).strip()
    text = ABBR_NAME_RE.sub(lambda m: ABBR_NAME[m.group(1).lower()], text)
    return WHITESPACE_RE.sub(" ", text).strip()


def norm_ext_addr_str(text: str) -> str:
    """Basic normalization + diacritic fold + '&' -> 'and' + address abbreviation expansion + placeholder removal."""
    if not text or pd.isna(text):
        return ""
    text = str(text).lower().replace("&", " and ")
    text = fold_diacritics_str(text)
    text = PUNCT_RE.sub(" ", text)
    text = PLACEHOLDER_ADDR_RE.sub(" ", text)
    text = WHITESPACE_RE.sub(" ", text).strip()
    text = ABBR_ADDR_RE.sub(lambda m: ABBR_ADDRESS[m.group(1).lower()], text)
    return WHITESPACE_RE.sub(" ", text).strip()


def sort_tokens_str(text: str) -> str:
    """Sort space-delimited tokens in a string."""
    if not text:
        return ""
    tokens = text.split()
    tokens.sort()
    return " ".join(tokens)


def extract_house_number(addr: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract first house/building number from address.
    Returns:
        (raw_first_number, leading_zeros_stripped_number) or (None, None)
    """
    if not addr or pd.isna(addr):
        return None, None
    m = HOUSE_NUM_RE.search(str(addr))
    if not m:
        return None, None
    raw = m.group(1)
    norm = raw.lstrip("0") or "0"
    return raw, norm


def extract_legal_form(name: str) -> Optional[str]:
    """
    Detect canonical legal form in normalized business name.
    """
    if not name:
        return None
    tokens = name.lower().split()
    n = len(tokens)
    # Check 2-token forms first (e.g. 'pvt ltd')
    for i in range(n - 1):
        bigram = f"{tokens[i]} {tokens[i+1]}"
        if bigram in LEGAL_FORMS_CANONICAL:
            return LEGAL_FORMS_CANONICAL[bigram]
    # Check 1-token forms
    for t in tokens:
        if t in LEGAL_FORMS_CANONICAL:
            return LEGAL_FORMS_CANONICAL[t]
    return None


def get_core_name(name_ext: str) -> str:
    """
    Strip legal form tokens to get core business name.
    """
    if not name_ext:
        return ""
    tokens = [t for t in name_ext.split() if t not in LEGAL_FORMS_CANONICAL]
    return " ".join(tokens).strip()


def extract_postal_code(addr: str, country: Optional[str] = None) -> Optional[str]:
    """Extract postal code / PIN / postcode based on country."""
    if not addr or pd.isna(addr):
        return None
    addr_str = str(addr)
    country_upper = (country or "").upper()
    if country_upper == "INDIA":
        m = POSTAL_INDIA_RE.search(addr_str)
        if m:
            return m.group(0).replace(" ", "")
    elif country_upper == "FRANCE":
        m = POSTAL_FRANCE_RE.search(addr_str)
        if m:
            return m.group(0)
    elif country_upper == "US":
        m = POSTAL_US_RE.search(addr_str)
        if m:
            return m.group(0)[:5]
    else:
        # Generic fallback
        m = POSTAL_INDIA_RE.search(addr_str) or POSTAL_US_RE.search(addr_str)
        if m:
            return m.group(0).replace(" ", "")
    return None


def extract_us_state_code(addr: str) -> Optional[str]:
    """Extract 2-letter US state code if detected."""
    if not addr or pd.isna(addr):
        return None
    m = US_STATE_RE.search(str(addr))
    if m:
        token = m.group(1).lower()
        return US_STATE_MAP.get(token)
    return None


def has_indic_script(text: str) -> bool:
    """Check if text contains Indic / native scripts."""
    if not text or pd.isna(text):
        return False
    return bool(INDIC_RE.search(str(text)))
