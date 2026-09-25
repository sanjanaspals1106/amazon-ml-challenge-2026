"""
Unit tests for M2 matching model, training, and prediction pipeline.
"""

import os
import tempfile
import numpy as np
import pandas as pd
import pytest

from src.features.pair_features import build_features
from src.model.labels import build_labels
from src.model.matcher import EntityResolutionMatcher
from src.model.prediction import predict_match_scores
from src.model.training import (
    evaluate_entity_macro_metrics,
    run_threshold_grid_search,
    split_grouped_by_s1,
    train_matcher,
)


@pytest.fixture
def synthetic_dataset():
    # 5 S1 entities
    s1_df = pd.DataFrame([
        {"entity_id": "S1-1", "business_name": "Google LLC", "business_address": "1600 Amphitheatre Pkwy, Mountain View, CA", "country": "US"},
        {"entity_id": "S1-2", "business_name": "Microsoft Corp", "business_address": "One Microsoft Way, Redmond, WA", "country": "US"},
        {"entity_id": "S1-3", "business_name": "Reliance Retail Ltd", "business_address": "Maker Chambers IV, Nariman Point, Mumbai", "country": "India"},
        {"entity_id": "S1-4", "business_name": "Tata Consultancy Services", "business_address": "Bandra Kurla Complex, Mumbai", "country": "India"},
        {"entity_id": "S1-5", "business_name": "Lone Singleton LLC", "business_address": "100 Desert Rd, Phoenix, AZ", "country": "US"},
    ])

    # Source records
    src_df = pd.DataFrame([
        {"entity_id": "S2-1", "business_name": "Google LLC", "business_address": "1600 Amphitheatre Pkwy, Mountain View, CA", "country": "US"},
        {"entity_id": "S2-2", "business_name": "Googler Tech", "business_address": "1600 Amphitheatre Pkwy, Mountain View, CA", "country": "US"},  # Decoy
        {"entity_id": "S2-3", "business_name": "Microsoft Corporation", "business_address": "1 Microsoft Way, Redmond, WA", "country": "US"},
        {"entity_id": "S3-4", "business_name": "Reliance Retail Limited", "business_address": "Maker Chambers, Nariman Point, Mumbai", "country": "India"},
        {"entity_id": "S3-5", "business_name": "Tata Sons", "business_address": "Bombay House, Mumbai", "country": "India"},  # False candidate
        {"entity_id": "S2-6", "business_name": "Tata Consultancy Services Ltd", "business_address": "BKC, Mumbai", "country": "India"},
        {"entity_id": "S2-7", "business_name": "Random Other Co", "business_address": "500 Elm St", "country": "US"},
    ])

    # Candidate pairs (M1 retrieval output simulation)
    cand_df = pd.DataFrame([
        {"s1_id": "S1-1", "source_record_id": "S2-1", "source": "S2", "country": "US", "retrieval_rank": 1, "retrieval_score": 0.99, "retrieval_method": "exact"},
        {"s1_id": "S1-1", "source_record_id": "S2-2", "source": "S2", "country": "US", "retrieval_rank": 2, "retrieval_score": 0.75, "retrieval_method": "tfidf"},
        {"s1_id": "S1-2", "source_record_id": "S2-3", "source": "S2", "country": "US", "retrieval_rank": 1, "retrieval_score": 0.95, "retrieval_method": "exact"},
        {"s1_id": "S1-3", "source_record_id": "S3-4", "source": "S3", "country": "India", "retrieval_rank": 1, "retrieval_score": 0.92, "retrieval_method": "exact"},
        {"s1_id": "S1-4", "source_record_id": "S3-5", "source": "S3", "country": "India", "retrieval_rank": 1, "retrieval_score": 0.60, "retrieval_method": "prefix"},
        {"s1_id": "S1-4", "source_record_id": "S2-6", "source": "S2", "country": "India", "retrieval_rank": 2, "retrieval_score": 0.88, "retrieval_method": "tfidf"},
        {"s1_id": "S1-5", "source_record_id": "S2-7", "source": "S2", "country": "US", "retrieval_rank": 1, "retrieval_score": 0.30, "retrieval_method": "prefix"},
    ])

    # Ground truth: S1-1 -> S2-1; S1-2 -> S2-3; S1-3 -> S3-4; S1-4 -> S2-6; S1-5 -> None (singleton)
    gt_df = pd.DataFrame([
        {"source1_entity_id": "S1-1", "matched_entity_ids": "S2-1"},
        {"source1_entity_id": "S1-2", "matched_entity_ids": "S2-3"},
        {"source1_entity_id": "S1-3", "matched_entity_ids": "S3-4"},
        {"source1_entity_id": "S1-4", "matched_entity_ids": "S2-6"},
        {"source1_entity_id": "S1-5", "matched_entity_ids": ""},
    ])

    return cand_df, s1_df, src_df, gt_df


def test_end_to_end_training_and_prediction(synthetic_dataset):
    cand_df, s1_df, src_df, gt_df = synthetic_dataset

    # 1. Feature extraction
    features_df = build_features(cand_df, s1_df, src_df)
    assert len(features_df) == len(cand_df)
    assert "n_token_sort_ratio" in features_df.columns
    assert "a_token_set_ratio" in features_df.columns

    # 2. Labels
    y = build_labels(cand_df, gt_df)
    assert len(y) == len(cand_df)
    # S1-1: S2-1 (1), S2-2 (0)
    # S1-2: S2-3 (1)
    # S1-3: S3-4 (1)
    # S1-4: S3-5 (0), S2-6 (1)
    # S1-5: S2-7 (0)
    assert list(y) == [1, 0, 1, 1, 0, 1, 0]

    # 3. Train model (XGBoost)
    config = {
        "model_type": "xgboost",
        "model_params": {
            "n_estimators": 20,
            "max_depth": 3,
            "learning_rate": 0.1,
            "min_child_weight": 0,
            "random_state": 42
        }
    }
    model = train_matcher(features_df, y, config=config)
    assert isinstance(model, EntityResolutionMatcher)

    # 4. Predict probabilities
    scores = predict_match_scores(model, features_df)
    assert len(scores) == len(cand_df)
    assert np.all((scores >= 0.0) & (scores <= 1.0))

    # High match pairs should score higher than low match pairs
    assert scores[0] > scores[1]  # S1-1: S2-1 vs S2-2 (decoy)
    assert scores[5] > scores[4]  # S1-4: S2-6 (true match) vs S3-5 (false candidate)

    # 5. Threshold evaluation
    best_res, res_df = run_threshold_grid_search(features_df, scores, gt_df, thresholds=[0.3, 0.5, 0.7])
    assert "macro_f05" in best_res
    assert best_res["macro_f05"] > 0.0
    assert len(res_df) == 3


def test_grouped_s1_split_no_leakage(synthetic_dataset):
    cand_df, s1_df, src_df, _ = synthetic_dataset
    feats = build_features(cand_df, s1_df, src_df)

    train_df, val_df = split_grouped_by_s1(feats, s1_col="s1_id", val_fraction=0.4, random_state=42)

    train_s1 = set(train_df["s1_id"].unique())
    val_s1 = set(val_df["s1_id"].unique())

    # Strictly 0 intersection!
    assert len(train_s1 & val_s1) == 0
    assert len(train_s1) + len(val_s1) == feats["s1_id"].nunique()


def test_hist_gradient_boosting_fallback(synthetic_dataset):
    cand_df, s1_df, src_df, gt_df = synthetic_dataset
    features_df = build_features(cand_df, s1_df, src_df)
    y = build_labels(cand_df, gt_df)

    config = {
        "model_type": "hist_gradient_boosting",
        "model_params": {
            "n_estimators": 15,
            "max_depth": 3,
            "learning_rate": 0.1,
            "random_state": 42
        }
    }
    model = train_matcher(features_df, y, config=config)
    assert model.model_type == "hist_gradient_boosting"

    scores = predict_match_scores(model, features_df)
    assert len(scores) == len(cand_df)
    assert np.all((scores >= 0.0) & (scores <= 1.0))


def test_model_save_and_load(synthetic_dataset):
    cand_df, s1_df, src_df, gt_df = synthetic_dataset
    features_df = build_features(cand_df, s1_df, src_df)
    y = build_labels(cand_df, gt_df)

    config = {"model_type": "xgboost", "model_params": {"n_estimators": 10, "max_depth": 2, "random_state": 42}}
    model = train_matcher(features_df, y, config=config)
    scores_orig = predict_match_scores(model, features_df)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "matcher.joblib")
        model.save(path)
        assert os.path.exists(path)

        loaded_model = EntityResolutionMatcher.load(path)
        scores_loaded = predict_match_scores(loaded_model, features_df)
        np.testing.assert_allclose(scores_orig, scores_loaded, rtol=1e-5)
