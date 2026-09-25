"""
Unit tests for M2 pair feature extraction.
"""

import numpy as np
import pandas as pd
import pytest

from src.features.pair_features import build_features
from src.features.normalization import (
    extract_house_number,
    extract_legal_form,
    extract_postal_code,
    extract_us_state_code,
    norm_basic_str,
    norm_ext_addr_str,
    norm_ext_name_str,
)


def _make_dfs(cand_rows, s1_rows, src_rows):
    cand_df = pd.DataFrame(cand_rows)
    s1_df = pd.DataFrame(s1_rows)
    src_df = pd.DataFrame(src_rows)
    return cand_df, s1_df, src_df


def test_identical_names():
    cand, s1, s2 = _make_dfs(
        [{"s1_id": "S1-1", "source_record_id": "S2-1", "source": "S2", "country": "US"}],
        [{"entity_id": "S1-1", "business_name": "Google LLC", "business_address": "1600 Amphitheatre Pkwy, Mountain View, CA", "country": "US"}],
        [{"entity_id": "S2-1", "business_name": "Google LLC", "business_address": "1600 Amphitheatre Pkwy, Mountain View, CA", "country": "US"}]
    )
    feats = build_features(cand, s1, s2)
    assert len(feats) == 1
    row = feats.iloc[0]
    assert row["n_token_sort_ratio"] == 1.0
    assert row["n_exact_ext_eq"] == 1.0
    assert row["n_token_jaccard"] == 1.0
    assert row["a_exact_ext_eq"] == 1.0
    assert row["a_house_num_exact_eq"] == 1.0


def test_reordered_names():
    cand, s1, s2 = _make_dfs(
        [{"s1_id": "S1-1", "source_record_id": "S2-1", "source": "S2", "country": "US"}],
        [{"entity_id": "S1-1", "business_name": "Baker Joe Bakery", "business_address": "10 Pine St", "country": "US"}],
        [{"entity_id": "S2-1", "business_name": "Joe Baker Bakery", "business_address": "10 Pine St", "country": "US"}]
    )
    feats = build_features(cand, s1, s2)
    row = feats.iloc[0]
    assert row["n_token_sort_ratio"] == 1.0
    assert row["n_exact_sorted_eq"] == 1.0
    assert row["n_token_jaccard"] == 1.0
    assert row["n_exact_basic_eq"] == 0.0


def test_punctuation_differences():
    cand, s1, s2 = _make_dfs(
        [{"s1_id": "S1-1", "source_record_id": "S2-1", "source": "S2", "country": "US"}],
        [{"entity_id": "S1-1", "business_name": "AT&T Mobility, Inc.", "business_address": "102 Main Rd.", "country": "US"}],
        [{"entity_id": "S2-1", "business_name": "AT and T Mobility Inc", "business_address": "102 Main Road", "country": "US"}]
    )
    feats = build_features(cand, s1, s2)
    row = feats.iloc[0]
    assert row["n_token_sort_ratio"] == 1.0
    assert row["n_exact_ext_eq"] == 1.0
    assert row["a_token_sort_ratio"] == 1.0
    assert row["a_exact_ext_eq"] == 1.0


def test_abbreviation_and_legal_forms():
    cand, s1, s2 = _make_dfs(
        [{"s1_id": "S1-1", "source_record_id": "S2-1", "source": "S2", "country": "US"}],
        [{"entity_id": "S1-1", "business_name": "Apex Corp", "business_address": "500 First Ave", "country": "US"}],
        [{"entity_id": "S2-1", "business_name": "Apex Corporation", "business_address": "500 1st Avenue", "country": "US"}]
    )
    feats = build_features(cand, s1, s2)
    row = feats.iloc[0]
    assert row["n_legal_form_s1_has"] == 1.0
    assert row["n_legal_form_source_has"] == 1.0
    assert row["n_legal_form_agree"] == 1.0
    assert row["n_legal_form_conflict"] == 0.0
    assert row["n_core_exact_eq"] == 1.0


def test_legal_form_conflict():
    cand, s1, s2 = _make_dfs(
        [{"s1_id": "S1-1", "source_record_id": "S2-1", "source": "S2", "country": "US"}],
        [{"entity_id": "S1-1", "business_name": "Apex Inc", "business_address": "500 First Ave", "country": "US"}],
        [{"entity_id": "S2-1", "business_name": "Apex Ltd", "business_address": "500 First Ave", "country": "US"}]
    )
    feats = build_features(cand, s1, s2)
    row = feats.iloc[0]
    assert row["n_legal_form_agree"] == 0.0
    assert row["n_legal_form_conflict"] == 1.0


def test_empty_addresses_never_crash():
    cand, s1, s2 = _make_dfs(
        [
            {"s1_id": "S1-1", "source_record_id": "S2-1", "source": "S2", "country": "India"},
            {"s1_id": "S1-2", "source_record_id": "S2-2", "source": "S2", "country": "India"}
        ],
        [
            {"entity_id": "S1-1", "business_name": "Ramesh Stores", "business_address": "", "country": "India"},
            {"entity_id": "S1-2", "business_name": "Suresh Trading", "business_address": None, "country": "India"}
        ],
        [
            {"entity_id": "S2-1", "business_name": "Ramesh Stores", "business_address": "Market Road, Delhi", "country": "India"},
            {"entity_id": "S2-2", "business_name": "Suresh Trading", "business_address": "", "country": "India"}
        ]
    )
    feats = build_features(cand, s1, s2)
    assert len(feats) == 2
    row0 = feats.iloc[0]
    assert row0["a_empty_s1"] == 1.0
    assert row0["a_empty_source"] == 0.0
    assert row0["a_empty_either"] == 1.0
    assert row0["a_token_set_ratio"] == 0.0

    row1 = feats.iloc[1]
    assert row1["a_empty_s1"] == 1.0
    assert row1["a_empty_source"] == 1.0
    assert row1["a_empty_either"] == 1.0
    assert row1["a_token_set_ratio"] == 0.0


def test_house_number_extraction_and_off_by_one():
    # Leading zeros normalization
    cand, s1, s2 = _make_dfs(
        [
            {"s1_id": "S1-1", "source_record_id": "S2-1", "source": "S2", "country": "US"},
            {"s1_id": "S1-2", "source_record_id": "S2-2", "source": "S2", "country": "US"}
        ],
        [
            {"entity_id": "S1-1", "business_name": "Test A", "business_address": "04 Oak St", "country": "US"},
            {"entity_id": "S1-2", "business_name": "Test B", "business_address": "15 Maple Ave", "country": "US"}
        ],
        [
            {"entity_id": "S2-1", "business_name": "Test A", "business_address": "4 Oak Street", "country": "US"},
            {"entity_id": "S2-2", "business_name": "Test B", "business_address": "16 Maple Ave", "country": "US"}
        ]
    )
    feats = build_features(cand, s1, s2)
    row0 = feats.iloc[0]
    assert row0["a_house_num_exact_eq"] == 0.0  # raw '04' != '4'
    assert row0["a_house_num_norm_eq"] == 1.0   # normalized '4' == '4'
    assert row0["a_house_num_off_by_one"] == 0.0

    row1 = feats.iloc[1]
    assert row1["a_house_num_exact_eq"] == 0.0
    assert row1["a_house_num_norm_eq"] == 0.0
    assert row1["a_house_num_off_by_one"] == 1.0  # 15 vs 16 is off-by-one!


def test_country_and_source_handling():
    cand, s1, s2 = _make_dfs(
        [
            {"s1_id": "S1-1", "source_record_id": "S2-1", "source": "S2", "country": "US"},
            {"s1_id": "S1-2", "source_record_id": "S3-2", "source": "S3", "country": "India"},
            {"s1_id": "S1-3", "source_record_id": "S2-3", "source": "S2", "country": "France"}
        ],
        [
            {"entity_id": "S1-1", "business_name": "US Entity", "business_address": "Austin TX 78701", "country": "US"},
            {"entity_id": "S1-2", "business_name": "India Entity", "business_address": "Connaught Place, New Delhi 110001", "country": "India"},
            {"entity_id": "S1-3", "business_name": "France Entity", "business_address": "10 Rue de Paris, 75001 Paris", "country": "France"}
        ],
        [
            {"entity_id": "S2-1", "business_name": "US Entity", "business_address": "Austin TX 78701", "country": "US"},
            {"entity_id": "S3-2", "business_name": "India Entity", "business_address": "Connaught Place, New Delhi 110001", "country": "India"},
            {"entity_id": "S2-3", "business_name": "France Entity", "business_address": "10 Rue de Paris, 75001 Paris", "country": "France"}
        ]
    )
    feats = build_features(cand, s1, s2)
    assert feats.iloc[0]["c_country_is_us"] == 1.0
    assert feats.iloc[0]["c_source_is_s2"] == 1.0

    assert feats.iloc[1]["c_country_is_india"] == 1.0
    assert feats.iloc[1]["c_source_is_s3"] == 1.0

    assert feats.iloc[2]["c_country_is_france"] == 1.0
    assert feats.iloc[2]["c_source_is_s2"] == 1.0


def test_initials_matching():
    cand, s1, s2 = _make_dfs(
        [{"s1_id": "S1-1", "source_record_id": "S2-1", "source": "S2", "country": "US"}],
        [{"entity_id": "S1-1", "business_name": "ibm", "business_address": "Armonk NY", "country": "US"}],
        [{"entity_id": "S2-1", "business_name": "International Business Machines", "business_address": "Armonk NY", "country": "US"}]
    )
    feats = build_features(cand, s1, s2)
    assert feats.iloc[0]["n_initials_match"] == 1.0
