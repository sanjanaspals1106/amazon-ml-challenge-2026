"""
M2 Model package: Matcher, Training, Prediction, and Labels.
"""

from src.model.labels import attach_labels, build_ground_truth_lookup, build_labels
from src.model.matcher import EntityResolutionMatcher
from src.model.prediction import predict_match_scores, score_candidates
from src.model.training import (
    evaluate_entity_macro_metrics,
    evaluate_segments,
    run_threshold_grid_search,
    split_grouped_by_s1,
    train_matcher,
)

__all__ = [
    "EntityResolutionMatcher",
    "train_matcher",
    "predict_match_scores",
    "score_candidates",
    "build_labels",
    "attach_labels",
    "build_ground_truth_lookup",
    "split_grouped_by_s1",
    "evaluate_entity_macro_metrics",
    "run_threshold_grid_search",
    "evaluate_segments",
]
