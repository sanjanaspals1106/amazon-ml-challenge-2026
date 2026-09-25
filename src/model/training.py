"""
Model training, grouped validation, and threshold evaluation for M2.

Validation strictly groups by S1 entity (s1_id) to prevent data leakage.
Computes competition-exact macro F0.5 (averaging per-entity F0.5 across all S1 entities,
including singletons).
"""

import logging
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from src.model.labels import build_ground_truth_lookup
from src.model.matcher import EntityResolutionMatcher, get_feature_columns

logger = logging.getLogger(__name__)


def compute_entity_f05(true_set: Set[str], pred_set: Set[str]) -> Tuple[float, float, float]:
    """
    Compute F0.5, Precision, and Recall for a single S1 entity according to competition criteria:
    - If true_set is empty (singleton):
        - 1.0 if pred_set is empty
        - 0.0 if pred_set has any predictions
    - If true_set is not empty:
        - 0.0 if pred_set is empty
        - (1.25 * P * R) / (0.25 * P + R) otherwise
    """
    if len(true_set) == 0:
        if len(pred_set) == 0:
            return 1.0, 1.0, 1.0
        else:
            return 0.0, 0.0, 1.0

    if len(pred_set) == 0:
        return 0.0, 1.0, 0.0

    tp = len(true_set & pred_set)
    precision = tp / len(pred_set)
    recall = tp / len(true_set)
    denom = 0.25 * precision + recall
    f05 = (1.25 * precision * recall) / denom if denom > 0 else 0.0
    return f05, precision, recall


def evaluate_entity_macro_metrics(
    eval_df: pd.DataFrame,
    scores: np.ndarray,
    threshold: float,
    ground_truth_lookup: Dict[str, Set[str]],
    s1_col: str = "s1_id",
    candidate_col: str = "source_record_id"
) -> Dict[str, Any]:
    """
    Evaluate macro F0.5, Precision, and Recall across all S1 entities in eval_df.
    """
    # Group candidates and predictions by s1_id
    preds_by_s1: Dict[str, Set[str]] = {}
    unique_s1s = eval_df[s1_col].unique()
    for s1 in unique_s1s:
        preds_by_s1[s1] = set()

    s1_vals = eval_df[s1_col].values
    cand_vals = eval_df[candidate_col].values

    for s1, cand, score in zip(s1_vals, cand_vals, scores):
        if score >= threshold:
            preds_by_s1[s1].add(cand)

    f05_list: List[float] = []
    p_list: List[float] = []
    r_list: List[float] = []

    # Segment metrics
    singleton_f05: List[float] = []
    non_singleton_f05: List[float] = []

    for s1 in unique_s1s:
        t_set = ground_truth_lookup.get(s1, set())
        p_set = preds_by_s1[s1]
        f, p, r = compute_entity_f05(t_set, p_set)
        f05_list.append(f)
        p_list.append(p)
        r_list.append(r)

        if len(t_set) == 0:
            singleton_f05.append(f)
        else:
            non_singleton_f05.append(f)

    macro_f05 = float(np.mean(f05_list)) if f05_list else 0.0
    macro_p = float(np.mean(p_list)) if p_list else 0.0
    macro_r = float(np.mean(r_list)) if r_list else 0.0

    res = {
        "threshold": threshold,
        "macro_f05": round(macro_f05, 4),
        "macro_precision": round(macro_p, 4),
        "macro_recall": round(macro_r, 4),
        "singleton_f05": round(float(np.mean(singleton_f05)), 4) if singleton_f05 else 0.0,
        "non_singleton_f05": round(float(np.mean(non_singleton_f05)), 4) if non_singleton_f05 else 0.0,
        "num_eval_s1_entities": len(unique_s1s),
    }

    # Pair-level metrics
    pair_labels = np.array([1 if cand_vals[i] in ground_truth_lookup.get(s1_vals[i], set()) else 0 for i in range(len(eval_df))])
    pair_preds = (scores >= threshold).astype(int)
    tp = int(((pair_preds == 1) & (pair_labels == 1)).sum())
    fp = int(((pair_preds == 1) & (pair_labels == 0)).sum())
    fn = int(((pair_preds == 0) & (pair_labels == 1)).sum())
    pair_prec = tp / max(1, tp + fp)
    pair_rec = tp / max(1, tp + fn)
    pair_f05 = (1.25 * pair_prec * pair_rec) / (0.25 * pair_prec + pair_rec) if (0.25 * pair_prec + pair_rec) > 0 else 0.0

    res.update({
        "pair_f05": round(pair_f05, 4),
        "pair_precision": round(pair_prec, 4),
        "pair_recall": round(pair_rec, 4),
        "pair_tp": tp,
        "pair_fp": fp,
        "pair_fn": fn,
    })

    return res


def split_grouped_by_s1(
    feature_df: pd.DataFrame,
    s1_col: str = "s1_id",
    val_fraction: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split feature_df into train and validation sets strictly grouped by S1 entity.
    Guarantees no S1 entity appears in both train and validation.
    """
    unique_s1 = feature_df[s1_col].unique()
    rng = np.random.default_rng(random_state)
    shuffled_s1 = rng.permutation(unique_s1)

    n_val = max(1, int(len(shuffled_s1) * val_fraction))
    val_s1_set = set(shuffled_s1[:n_val])

    is_val = feature_df[s1_col].isin(val_s1_set)
    train_df = feature_df[~is_val].copy().reset_index(drop=True)
    val_df = feature_df[is_val].copy().reset_index(drop=True)

    logger.info(f"Grouped S1 split: Train={len(train_df)} pairs ({len(unique_s1) - n_val} S1s), Val={len(val_df)} pairs ({n_val} S1s)")
    return train_df, val_df


def train_matcher(
    X_train: Union[pd.DataFrame, np.ndarray],
    y_train: Union[pd.Series, np.ndarray],
    config: Optional[Dict[str, Any]] = None
) -> EntityResolutionMatcher:
    """
    Train an EntityResolutionMatcher model.

    Args:
        X_train: Feature DataFrame or array.
        y_train: Target binary labels (1=match, 0=non-match).
        config: Model and training hyperparameters.

    Returns:
        Trained EntityResolutionMatcher instance.
    """
    matcher = EntityResolutionMatcher(config=config)
    matcher.fit(X_train, y_train)
    return matcher


def run_threshold_grid_search(
    val_df: pd.DataFrame,
    val_scores: np.ndarray,
    ground_truth: Union[pd.DataFrame, Dict[str, Set[str]]],
    thresholds: Optional[List[float]] = None,
    s1_col: str = "s1_id",
    candidate_col: str = "source_record_id"
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Evaluate a grid of match score thresholds on validation set.

    Args:
        val_df: Validation candidate pairs with metadata.
        val_scores: Predicted probabilities.
        ground_truth: Ground truth DataFrame or lookup dict.
        thresholds: List of thresholds to evaluate.

    Returns:
        (best_result_dict, all_results_dataframe)
    """
    if thresholds is None:
        thresholds = [0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]

    gt_lookup = build_ground_truth_lookup(ground_truth) if isinstance(ground_truth, pd.DataFrame) else ground_truth

    results = []
    best_f05 = -1.0
    best_res: Dict[str, Any] = {}

    for t in thresholds:
        m = evaluate_entity_macro_metrics(
            eval_df=val_df,
            scores=val_scores,
            threshold=t,
            ground_truth_lookup=gt_lookup,
            s1_col=s1_col,
            candidate_col=candidate_col
        )
        results.append(m)
        if m["macro_f05"] > best_f05:
            best_f05 = m["macro_f05"]
            best_res = m

    res_df = pd.DataFrame(results)
    return best_res, res_df


def evaluate_segments(
    val_df: pd.DataFrame,
    val_scores: np.ndarray,
    ground_truth: Union[pd.DataFrame, Dict[str, Set[str]]],
    threshold: float,
    s1_col: str = "s1_id",
    candidate_col: str = "source_record_id"
) -> Dict[str, Any]:
    """
    Compute segment breakdowns (US vs India vs France, S2 vs S3).
    """
    gt_lookup = build_ground_truth_lookup(ground_truth) if isinstance(ground_truth, pd.DataFrame) else ground_truth
    df = val_df.copy()
    df["_score"] = val_scores

    segment_results: Dict[str, Any] = {}

    # Country breakdown
    if "country" in df.columns:
        for c in df["country"].dropna().unique():
            c_mask = df["country"] == c
            sub = df[c_mask].reset_index(drop=True)
            if len(sub) > 0:
                m = evaluate_entity_macro_metrics(
                    eval_df=sub,
                    scores=sub["_score"].values,
                    threshold=threshold,
                    ground_truth_lookup=gt_lookup,
                    s1_col=s1_col,
                    candidate_col=candidate_col
                )
                segment_results[f"country_{c}"] = m

    # Source breakdown (S2 vs S3)
    if "source" in df.columns:
        for s in df["source"].dropna().unique():
            s_mask = df["source"] == s
            sub = df[s_mask].reset_index(drop=True)
            if len(sub) > 0:
                m = evaluate_entity_macro_metrics(
                    eval_df=sub,
                    scores=sub["_score"].values,
                    threshold=threshold,
                    ground_truth_lookup=gt_lookup,
                    s1_col=s1_col,
                    candidate_col=candidate_col
                )
                segment_results[f"source_{s}"] = m

    return segment_results
