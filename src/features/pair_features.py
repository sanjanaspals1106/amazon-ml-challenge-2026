"""
Deterministic pairwise feature extraction for candidate entity pairs (M2).

Contract:
    build_features(
        candidate_df: pd.DataFrame,
        s1_df: pd.DataFrame,
        source_df: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        config: Optional[Dict] = None
    ) -> pd.DataFrame
"""

import logging
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein

from src.features.normalization import (
    extract_house_number,
    extract_legal_form,
    extract_postal_code,
    extract_us_state_code,
    fold_diacritics_str,
    get_core_name,
    has_indic_script,
    norm_basic_str,
    norm_ext_addr_str,
    norm_ext_name_str,
    sort_tokens_str,
)

logger = logging.getLogger(__name__)


class PrecomputedEntity:
    """Precomputed representations of a single entity (name + address) to avoid repeated recomputation."""
    __slots__ = (
        "raw_name", "raw_addr", "country",
        "name_basic", "name_ext", "name_sorted", "name_tokens", "name_token_set",
        "name_core", "name_legal_form", "has_indic", "name_initials",
        "addr_basic", "addr_ext", "addr_sorted", "addr_tokens", "addr_token_set",
        "is_addr_empty", "house_raw", "house_norm", "postal_code", "us_state"
    )

    def __init__(self, raw_name: Any, raw_addr: Any, country: Any):
        r_name = "" if pd.isna(raw_name) else str(raw_name).strip()
        r_addr = "" if pd.isna(raw_addr) else str(raw_addr).strip()
        cntry = "" if pd.isna(country) else str(country).strip()

        self.raw_name = r_name
        self.raw_addr = r_addr
        self.country = cntry

        # Name normalization
        nb = norm_basic_str(r_name)
        ne = norm_ext_name_str(r_name)
        self.name_basic = nb
        self.name_ext = ne
        self.name_sorted = sort_tokens_str(ne)
        tokens = ne.split()
        self.name_tokens = tokens
        self.name_token_set = set(tokens)
        self.name_core = get_core_name(ne)
        self.name_legal_form = extract_legal_form(ne)
        self.has_indic = has_indic_script(r_name) or has_indic_script(r_addr)
        self.name_initials = "".join([t[0] for t in tokens if t]) if tokens else ""

        # Address normalization
        ab = norm_basic_str(r_addr)
        ae = norm_ext_addr_str(r_addr)
        self.addr_basic = ab
        self.addr_ext = ae
        self.addr_sorted = sort_tokens_str(ae)
        a_tokens = ae.split()
        self.addr_tokens = a_tokens
        self.addr_token_set = set(a_tokens)
        self.is_addr_empty = (len(ae) == 0)

        # Address structural components
        h_raw, h_norm = extract_house_number(r_addr)
        self.house_raw = h_raw
        self.house_norm = h_norm
        self.postal_code = extract_postal_code(r_addr, cntry)
        self.us_state = extract_us_state_code(r_addr) if cntry.upper() == "US" else None


def _precompute_entities(df: pd.DataFrame, id_col: str) -> Dict[str, PrecomputedEntity]:
    """Precompute all unique entities in a dataframe for fast feature extraction."""
    lookup: Dict[str, PrecomputedEntity] = {}
    if df is None or len(df) == 0:
        return lookup

    name_col = "business_name" if "business_name" in df.columns else df.columns[1]
    addr_col = "business_address" if "business_address" in df.columns else df.columns[2]
    cntry_col = "country" if "country" in df.columns else None

    # Vectorized / iterrows optimization
    ids = df[id_col].values
    names = df[name_col].values
    addrs = df[addr_col].values
    countries = df[cntry_col].values if cntry_col else [None] * len(df)

    for eid, name, addr, cntry in zip(ids, names, addrs, countries):
        lookup[str(eid)] = PrecomputedEntity(name, addr, cntry)

    return lookup


def _char_3gram_jaccard(s1: str, s2: str) -> float:
    """Character 3-gram Jaccard similarity between two strings."""
    s1_clean = s1.replace(" ", "")
    s2_clean = s2.replace(" ", "")
    if len(s1_clean) < 3 or len(s2_clean) < 3:
        return 1.0 if s1_clean == s2_clean and len(s1_clean) > 0 else 0.0
    g1 = {s1_clean[i:i + 3] for i in range(len(s1_clean) - 2)}
    g2 = {s2_clean[i:i + 3] for i in range(len(s2_clean) - 2)}
    u = len(g1 | g2)
    return len(g1 & g2) / u if u > 0 else 0.0


def _token_jaccard(set1: Set[str], set2: Set[str]) -> float:
    """Token Jaccard between two token sets."""
    u = len(set1 | set2)
    return len(set1 & set2) / u if u > 0 else 0.0


def _token_overlap(set1: Set[str], set2: Set[str]) -> float:
    """Token overlap: |intersection| / min(|s1|, |s2|)."""
    m = min(len(set1), len(set2))
    return len(set1 & set2) / m if m > 0 else 0.0


def _check_initials_match(e1: PrecomputedEntity, e2: PrecomputedEntity) -> float:
    """Check if one entity's name matches the acronym/initials of the other."""
    # Check if e1 is short acronym of e2
    if 2 <= len(e1.name_ext) <= 5 and e1.name_ext.isalpha():
        if e1.name_ext == e2.name_initials:
            return 1.0
    # Check if e2 is short acronym of e1
    if 2 <= len(e2.name_ext) <= 5 and e2.name_ext.isalpha():
        if e2.name_ext == e1.name_initials:
            return 1.0
    return 0.0


def _standardize_candidate_columns(candidate_df: pd.DataFrame) -> pd.DataFrame:
    """Map candidate column variants to canonical M2 names."""
    df = candidate_df.copy()
    col_map = {
        "source1_entity_id": "s1_id",
        "s2_or_s3_id": "source_record_id",
        "candidate_id": "source_record_id",
        "candidate_entity_id": "source_record_id",
    }
    for old_col, new_col in col_map.items():
        if old_col in df.columns and new_col not in df.columns:
            df.rename(columns={old_col: new_col}, inplace=True)

    if "s1_id" not in df.columns:
        raise ValueError(f"candidate_df missing required S1 ID column (expected 's1_id'). Found: {list(df.columns)}")
    if "source_record_id" not in df.columns:
        raise ValueError(f"candidate_df missing required candidate ID column (expected 'source_record_id'). Found: {list(df.columns)}")

    # Ensure source column exists (e.g. 'S2' or 'S3')
    if "source" not in df.columns:
        df["source"] = df["source_record_id"].astype(str).apply(
            lambda x: "S2" if x.startswith("S2") else ("S3" if x.startswith("S3") else "unknown")
        )

    # Context column defaults if not provided by M1
    if "retrieval_rank" not in df.columns:
        df["retrieval_rank"] = -1
    if "retrieval_score" not in df.columns:
        df["retrieval_score"] = 0.0
    if "retrieval_method" not in df.columns:
        df["retrieval_method"] = "unknown"

    return df


def build_features(
    candidate_df: pd.DataFrame,
    s1_df: pd.DataFrame,
    source_df: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
    config: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """
    Extract pairwise features for each candidate pair.

    Args:
        candidate_df: Candidate pairs from M1. Required cols: s1_id, source_record_id.
                      Optional cols: source, country, retrieval_rank, retrieval_score, retrieval_method.
        s1_df: Reference S1 entity records (entity_id, business_name, business_address, country).
        source_df: Source records (either a single combined DataFrame or a dict {'S2': s2_df, 'S3': s3_df}).
        config: Optional configuration dictionary.

    Returns:
        pd.DataFrame containing preserved identity columns and all computed numerical/categorical features.
    """
    if candidate_df is None or len(candidate_df) == 0:
        logger.warning("Empty candidate_df provided to build_features. Returning empty DataFrame.")
        return pd.DataFrame()

    cand = _standardize_candidate_columns(candidate_df)

    # 1. Precompute S1 entities
    s1_id_col = "entity_id" if "entity_id" in s1_df.columns else ("s1_id" if "s1_id" in s1_df.columns else s1_df.columns[0])
    s1_lookup = _precompute_entities(s1_df, s1_id_col)

    # 2. Precompute Source entities (S2 / S3)
    source_lookup: Dict[str, PrecomputedEntity] = {}
    if isinstance(source_df, dict):
        for src_name, sdf in source_df.items():
            src_id_col = "entity_id" if "entity_id" in sdf.columns else sdf.columns[0]
            source_lookup.update(_precompute_entities(sdf, src_id_col))
    elif isinstance(source_df, pd.DataFrame):
        src_id_col = "entity_id" if "entity_id" in source_df.columns else source_df.columns[0]
        source_lookup = _precompute_entities(source_df, src_id_col)
    else:
        raise TypeError(f"source_df must be pd.DataFrame or Dict[str, pd.DataFrame], got {type(source_df)}")

    # 3. Precompute contextual aggregation stats
    s1_counts = cand["s1_id"].value_counts().to_dict()
    comp_counts = cand["source_record_id"].value_counts().to_dict()

    # Precompute maximum retrieval score per s1_id to identify best candidate
    if "retrieval_score" in cand.columns:
        max_score_per_s1 = cand.groupby("s1_id")["retrieval_score"].max().to_dict()
    else:
        max_score_per_s1 = {}

    # 4. Extract features row by row using precomputed representations
    num_rows = len(cand)
    feature_rows: List[Dict[str, Any]] = []

    s1_ids = cand["s1_id"].astype(str).values
    src_ids = cand["source_record_id"].astype(str).values
    sources = cand["source"].astype(str).values
    countries = cand["country"].astype(str).values if "country" in cand.columns else [None] * num_rows
    ranks = pd.to_numeric(cand["retrieval_rank"], errors="coerce").fillna(-1).values
    scores = pd.to_numeric(cand["retrieval_score"], errors="coerce").fillna(0.0).values
    methods = cand["retrieval_method"].astype(str).values

    # Fallback dummy entity for missing entities
    dummy_entity = PrecomputedEntity("", "", "")

    for i in range(num_rows):
        s1_key = s1_ids[i]
        src_key = src_ids[i]
        src_tag = sources[i]
        rank_val = ranks[i]
        score_val = scores[i]
        method_val = methods[i]

        e1 = s1_lookup.get(s1_key, dummy_entity)
        e2 = source_lookup.get(src_key, dummy_entity)

        # Determine country
        cntry = countries[i] or e1.country or e2.country or "unknown"
        cntry_upper = cntry.upper()

        row: Dict[str, Any] = {
            "s1_id": s1_key,
            "source_record_id": src_key,
            "source": src_tag,
        }

        # ----------------- NAME FEATURES -----------------
        # Lexical Similarity
        row["n_token_sort_ratio"] = fuzz.token_sort_ratio(e1.name_ext, e2.name_ext) / 100.0
        row["n_token_set_ratio"] = fuzz.token_set_ratio(e1.name_ext, e2.name_ext) / 100.0
        row["n_partial_ratio"] = fuzz.partial_ratio(e1.name_ext, e2.name_ext) / 100.0
        row["n_levenshtein"] = Levenshtein.normalized_similarity(e1.name_ext, e2.name_ext)
        row["n_jaro_winkler"] = JaroWinkler.similarity(e1.name_ext, e2.name_ext)
        row["n_token_jaccard"] = _token_jaccard(e1.name_token_set, e2.name_token_set)
        row["n_token_overlap"] = _token_overlap(e1.name_token_set, e2.name_token_set)
        row["n_char_3gram_jaccard"] = _char_3gram_jaccard(e1.name_ext, e2.name_ext)

        # Structural & Token Differences
        first_t1 = e1.name_tokens[0] if e1.name_tokens else ""
        first_t2 = e2.name_tokens[0] if e2.name_tokens else ""
        row["n_first_token_eq"] = 1.0 if (first_t1 and first_t1 == first_t2) else 0.0
        row["n_token_count_diff"] = float(abs(len(e1.name_tokens) - len(e2.name_tokens)))
        row["n_char_length_diff"] = float(abs(len(e1.name_basic) - len(e2.name_basic)))
        row["n_exact_basic_eq"] = 1.0 if (e1.name_basic and e1.name_basic == e2.name_basic) else 0.0
        row["n_exact_ext_eq"] = 1.0 if (e1.name_ext and e1.name_ext == e2.name_ext) else 0.0
        row["n_exact_sorted_eq"] = 1.0 if (e1.name_sorted and e1.name_sorted == e2.name_sorted) else 0.0
        row["n_extra_tokens_count"] = float(abs(len(e1.name_token_set) - len(e2.name_token_set)))

        # Semantics & Legal Forms
        row["n_core_name_sim"] = fuzz.token_sort_ratio(e1.name_core, e2.name_core) / 100.0 if (e1.name_core and e2.name_core) else row["n_token_sort_ratio"]
        row["n_core_exact_eq"] = 1.0 if (e1.name_core and e1.name_core == e2.name_core) else 0.0
        
        lf1 = e1.name_legal_form
        lf2 = e2.name_legal_form
        row["n_legal_form_s1_has"] = 1.0 if lf1 else 0.0
        row["n_legal_form_source_has"] = 1.0 if lf2 else 0.0
        row["n_legal_form_agree"] = 1.0 if (lf1 and lf2 and lf1 == lf2) else 0.0
        row["n_legal_form_conflict"] = 1.0 if (lf1 and lf2 and lf1 != lf2) else 0.0
        row["n_initials_match"] = _check_initials_match(e1, e2)
        row["n_has_indic_script"] = 1.0 if (e1.has_indic or e2.has_indic) else 0.0

        # ----------------- ADDRESS FEATURES -----------------
        # Empty indicators
        row["a_empty_s1"] = 1.0 if e1.is_addr_empty else 0.0
        row["a_empty_source"] = 1.0 if e2.is_addr_empty else 0.0
        row["a_empty_either"] = 1.0 if (e1.is_addr_empty or e2.is_addr_empty) else 0.0

        if row["a_empty_either"] == 1.0:
            # Safe imputations for empty address: 0.0 for similarities, 0.0 for matches
            row["a_token_set_ratio"] = 0.0
            row["a_token_sort_ratio"] = 0.0
            row["a_levenshtein"] = 0.0
            row["a_token_jaccard"] = 0.0
            row["a_token_overlap"] = 0.0
            row["a_char_3gram_jaccard"] = 0.0
            row["a_exact_basic_eq"] = 0.0
            row["a_exact_ext_eq"] = 0.0
            row["a_exact_sorted_eq"] = 0.0
            row["a_char_length_diff"] = float(abs(len(e1.addr_basic) - len(e2.addr_basic)))
            row["a_token_count_diff"] = float(abs(len(e1.addr_tokens) - len(e2.addr_tokens)))
            row["a_house_num_both_present"] = 0.0
            row["a_house_num_exact_eq"] = 0.0
            row["a_house_num_norm_eq"] = 0.0
            row["a_house_num_off_by_one"] = 0.0
            row["a_postal_code_both_present"] = 0.0
            row["a_postal_code_eq"] = 0.0
            row["a_us_state_both_present"] = 0.0
            row["a_us_state_agree"] = 0.0
        else:
            # Address Similarities
            row["a_token_set_ratio"] = fuzz.token_set_ratio(e1.addr_ext, e2.addr_ext) / 100.0
            row["a_token_sort_ratio"] = fuzz.token_sort_ratio(e1.addr_ext, e2.addr_ext) / 100.0
            row["a_levenshtein"] = Levenshtein.normalized_similarity(e1.addr_ext, e2.addr_ext)
            row["a_token_jaccard"] = _token_jaccard(e1.addr_token_set, e2.addr_token_set)
            row["a_token_overlap"] = _token_overlap(e1.addr_token_set, e2.addr_token_set)
            row["a_char_3gram_jaccard"] = _char_3gram_jaccard(e1.addr_ext, e2.addr_ext)

            # Address Structural
            row["a_exact_basic_eq"] = 1.0 if (e1.addr_basic and e1.addr_basic == e2.addr_basic) else 0.0
            row["a_exact_ext_eq"] = 1.0 if (e1.addr_ext and e1.addr_ext == e2.addr_ext) else 0.0
            row["a_exact_sorted_eq"] = 1.0 if (e1.addr_sorted and e1.addr_sorted == e2.addr_sorted) else 0.0
            row["a_char_length_diff"] = float(abs(len(e1.addr_basic) - len(e2.addr_basic)))
            row["a_token_count_diff"] = float(abs(len(e1.addr_tokens) - len(e2.addr_tokens)))

            # House number comparisons
            h1_raw, h1_norm = e1.house_raw, e1.house_norm
            h2_raw, h2_norm = e2.house_raw, e2.house_norm
            both_h = (h1_raw is not None and h2_raw is not None)
            row["a_house_num_both_present"] = 1.0 if both_h else 0.0
            row["a_house_num_exact_eq"] = 1.0 if (both_h and h1_raw == h2_raw) else 0.0
            row["a_house_num_norm_eq"] = 1.0 if (both_h and h1_norm == h2_norm) else 0.0

            # Off-by-one check
            if both_h and h1_norm.isdigit() and h2_norm.isdigit():
                row["a_house_num_off_by_one"] = 1.0 if abs(int(h1_norm) - int(h2_norm)) == 1 else 0.0
            else:
                row["a_house_num_off_by_one"] = 0.0

            # Postal code comparisons
            p1, p2 = e1.postal_code, e2.postal_code
            both_p = (p1 is not None and p2 is not None)
            row["a_postal_code_both_present"] = 1.0 if both_p else 0.0
            row["a_postal_code_eq"] = 1.0 if (both_p and p1 == p2) else 0.0

            # US State comparisons
            st1, st2 = e1.us_state, e2.us_state
            both_st = (st1 is not None and st2 is not None)
            row["a_us_state_both_present"] = 1.0 if both_st else 0.0
            row["a_us_state_agree"] = 1.0 if (both_st and st1 == st2) else 0.0

        # ----------------- CONTEXT & RETRIEVAL FEATURES -----------------
        row["c_country_is_us"] = 1.0 if cntry_upper == "US" else 0.0
        row["c_country_is_india"] = 1.0 if cntry_upper == "INDIA" else 0.0
        row["c_country_is_france"] = 1.0 if cntry_upper == "FRANCE" else 0.0

        s_upper = src_tag.upper()
        row["c_source_is_s2"] = 1.0 if "S2" in s_upper or s_upper == "2" else 0.0
        row["c_source_is_s3"] = 1.0 if "S3" in s_upper or s_upper == "3" else 0.0

        row["c_retrieval_rank"] = float(rank_val)
        row["c_retrieval_score"] = float(score_val)
        row["c_is_top_1"] = 1.0 if rank_val == 1 else 0.0

        row["c_candidate_count_s1"] = float(s1_counts.get(s1_key, 1))
        row["c_candidate_competition_count"] = float(comp_counts.get(src_key, 1))

        best_score = max_score_per_s1.get(s1_key, score_val)
        row["c_is_best_score_for_s1"] = 1.0 if score_val >= best_score else 0.0

        feature_rows.append(row)

    feat_df = pd.DataFrame(feature_rows)
    return feat_df
