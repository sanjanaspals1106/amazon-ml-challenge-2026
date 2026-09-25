"""
Candidate Generation and Blocking Module for Amazon ML Challenge 2026.
Teammate A (Person 2).
"""

from .candidate_generator import CandidateGenerator
from .text_processor import (
    normalize_name,
    normalize_address,
    sort_tokens,
    extract_prefixes_4,
    extract_house_number,
)

__all__ = [
    "CandidateGenerator",
    "normalize_name",
    "normalize_address",
    "sort_tokens",
    "extract_prefixes_4",
    "extract_house_number",
]
