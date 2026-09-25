"""
Unit tests for M2 training label builder.
"""

import numpy as np
import pandas as pd
import pytest

from src.model.labels import attach_labels, build_ground_truth_lookup, build_labels, compute_label_summary


def test_positive_and_negative_labels():
    gt_df = pd.DataFrame([
        {"source1_entity_id": "S1-100", "matched_entity_ids": "S2-500,S3-600"},
        {"source1_entity_id": "S1-101", "matched_entity_ids": "S2-501"},
    ])

    cand_df = pd.DataFrame([
        {"s1_id": "S1-100", "source_record_id": "S2-500"},  # Positive
        {"s1_id": "S1-100", "source_record_id": "S3-600"},  # Positive
        {"s1_id": "S1-100", "source_record_id": "S2-999"},  # Negative (retrieved false candidate)
        {"s1_id": "S1-101", "source_record_id": "S2-501"},  # Positive
        {"s1_id": "S1-101", "source_record_id": "S2-888"},  # Negative
    ])

    labels = build_labels(cand_df, gt_df)
    assert np.array_equal(labels, [1, 1, 0, 1, 0])


def test_singleton_handling():
    # S1-200 is a singleton: has 0 matches in ground truth
    gt_df = pd.DataFrame([
        {"source1_entity_id": "S1-200", "matched_entity_ids": ""},
    ])

    cand_df = pd.DataFrame([
        {"s1_id": "S1-200", "source_record_id": "S2-111"},
        {"s1_id": "S1-200", "source_record_id": "S3-222"},
    ])

    labels = build_labels(cand_df, gt_df)
    # Both retrieved candidates should be labeled 0 (negative)
    assert np.array_equal(labels, [0, 0])


def test_multiple_positives_handling():
    # S1-300 matches 3 records
    gt_df = pd.DataFrame([
        {"source1_entity_id": "S1-300", "matched_entity_ids": "S2-1,S2-2,S3-3"},
    ])

    cand_df = pd.DataFrame([
        {"s1_id": "S1-300", "source_record_id": "S2-1"},
        {"s1_id": "S1-300", "source_record_id": "S2-2"},
        {"s1_id": "S1-300", "source_record_id": "S3-3"},
        {"s1_id": "S1-300", "source_record_id": "S2-4"},  # Negative
    ])

    labels = build_labels(cand_df, gt_df)
    assert np.array_equal(labels, [1, 1, 1, 0])


def test_attach_labels_and_summary():
    gt_df = pd.DataFrame([
        {"source1_entity_id": "S1-1", "matched_entity_ids": "S2-10"},
        {"source1_entity_id": "S1-2", "matched_entity_ids": ""},
    ])
    cand_df = pd.DataFrame([
        {"s1_id": "S1-1", "source_record_id": "S2-10"},
        {"s1_id": "S1-1", "source_record_id": "S2-11"},
        {"s1_id": "S1-2", "source_record_id": "S2-20"},
    ])

    labeled_df = attach_labels(cand_df, gt_df, label_col="target")
    assert "target" in labeled_df.columns
    assert list(labeled_df["target"]) == [1, 0, 0]

    summary = compute_label_summary(cand_df, labeled_df["target"].values, gt_df)
    assert summary["total_candidate_pairs"] == 3
    assert summary["positive_pairs"] == 1
    assert summary["negative_pairs"] == 2
    assert summary["ground_truth_singletons"] == 1
