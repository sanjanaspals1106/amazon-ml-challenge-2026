"""
Training label builder for candidate pairs using competition ground truth (M2).

Converts M1 candidate pairs into binary labels:
- 1 = candidate record is actually associated with that S1 entity in ground truth.
- 0 = candidate record was retrieved for that S1 entity but is not its true match.

Handles:
- S1 with 0 matches (singletons)
- S1 with 1 match
- S1 with multiple matches (both S2 and S3, or multiple S2/S3)
- Preserves hard negatives produced by blocking without artificial random negatives
"""

import logging
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def build_ground_truth_lookup(
    ground_truth_df: pd.DataFrame,
    s1_col: str = "source1_entity_id",
    matches_col: str = "matched_entity_ids"
) -> Dict[str, Set[str]]:
    """
    Build a fast lookup dictionary from ground truth DataFrame.
    
    Args:
        ground_truth_df: DataFrame with source1_entity_id and matched_entity_ids.
        s1_col: Name of S1 entity column.
        matches_col: Name of matched entity IDs column (comma-separated).

    Returns:
        Dict mapping s1_id -> set of true matched entity IDs.
    """
    # Accommodate alternative column names
    if s1_col not in ground_truth_df.columns:
        for candidate in ("s1_id", "entity_id", "source1_id"):
            if candidate in ground_truth_df.columns:
                s1_col = candidate
                break
    if matches_col not in ground_truth_df.columns:
        for candidate in ("matched_ids", "matches", "target_ids"):
            if candidate in ground_truth_df.columns:
                matches_col = candidate
                break

    lookup: Dict[str, Set[str]] = {}
    s1_vals = ground_truth_df[s1_col].astype(str).values
    match_vals = ground_truth_df[matches_col].fillna("").astype(str).values

    for s1_id, match_str in zip(s1_vals, match_vals):
        match_str = match_str.strip()
        if not match_str:
            lookup[s1_id] = set()
        else:
            ids = {m.strip() for m in match_str.split(",") if m.strip()}
            lookup[s1_id] = ids

    return lookup


def build_labels(
    candidate_df: pd.DataFrame,
    ground_truth: Union[pd.DataFrame, Dict[str, Set[str]]],
    s1_col: str = "s1_id",
    candidate_col: str = "source_record_id"
) -> np.ndarray:
    """
    Generate binary labels (1/0) for each pair in candidate_df.

    Args:
        candidate_df: DataFrame of candidate pairs.
        ground_truth: Either ground truth DataFrame or precomputed lookup dict.
        s1_col: Column name for S1 ID in candidate_df.
        candidate_col: Column name for candidate ID in candidate_df.

    Returns:
        1D numpy array of integer labels (1 for true match, 0 for negative candidate).
    """
    if s1_col not in candidate_df.columns:
        if "source1_entity_id" in candidate_df.columns:
            s1_col = "source1_entity_id"
        else:
            raise ValueError(f"candidate_df missing S1 ID column '{s1_col}'")

    if candidate_col not in candidate_df.columns:
        for alt in ("s2_or_s3_id", "candidate_id", "candidate_entity_id", "entity_id"):
            if alt in candidate_df.columns:
                candidate_col = alt
                break
        else:
            raise ValueError(f"candidate_df missing candidate ID column '{candidate_col}'")

    if isinstance(ground_truth, pd.DataFrame):
        lookup = build_ground_truth_lookup(ground_truth)
    elif isinstance(ground_truth, dict):
        lookup = ground_truth
    else:
        raise TypeError(f"ground_truth must be pd.DataFrame or Dict[str, Set[str]], got {type(ground_truth)}")

    s1_vals = candidate_df[s1_col].astype(str).values
    cand_vals = candidate_df[candidate_col].astype(str).values

    labels = np.zeros(len(candidate_df), dtype=np.int32)
    empty_set: Set[str] = set()

    for i in range(len(candidate_df)):
        true_set = lookup.get(s1_vals[i], empty_set)
        if cand_vals[i] in true_set:
            labels[i] = 1

    return labels


def attach_labels(
    candidate_df: pd.DataFrame,
    ground_truth: Union[pd.DataFrame, Dict[str, Set[str]]],
    label_col: str = "label",
    s1_col: str = "s1_id",
    candidate_col: str = "source_record_id"
) -> pd.DataFrame:
    """
    Attach binary labels as a column directly to candidate_df or feature_df.

    Returns:
        DataFrame with new label column.
    """
    df = candidate_df.copy()
    df[label_col] = build_labels(df, ground_truth, s1_col=s1_col, candidate_col=candidate_col)
    return df


def compute_label_summary(
    candidate_df: pd.DataFrame,
    labels: np.ndarray,
    ground_truth: Optional[Union[pd.DataFrame, Dict[str, Set[str]]]] = None,
    s1_col: str = "s1_id"
) -> Dict[str, Any]:
    """
    Compute diagnostics on candidate pairs and their labels.
    """
    total_pairs = len(labels)
    num_pos = int((labels == 1).sum())
    num_neg = int((labels == 0).sum())
    pos_rate = float(num_pos / total_pairs) if total_pairs > 0 else 0.0

    if s1_col not in candidate_df.columns and "source1_entity_id" in candidate_df.columns:
        s1_col = "source1_entity_id"

    num_unique_s1 = int(candidate_df[s1_col].nunique()) if s1_col in candidate_df.columns else 0

    summary = {
        "total_candidate_pairs": total_pairs,
        "positive_pairs": num_pos,
        "negative_pairs": num_neg,
        "positive_rate": round(pos_rate, 4),
        "unique_s1_entities": num_unique_s1,
    }

    if ground_truth is not None:
        lookup = build_ground_truth_lookup(ground_truth) if isinstance(ground_truth, pd.DataFrame) else ground_truth
        total_gt_matches = sum(len(matches) for matches in lookup.values())
        gt_singletons = sum(1 for matches in lookup.values() if len(matches) == 0)
        gt_multi = sum(1 for matches in lookup.values() if len(matches) > 1)

        summary.update({
            "total_ground_truth_matches": total_gt_matches,
            "candidate_recall_ceiling": round(num_pos / total_gt_matches, 4) if total_gt_matches > 0 else 0.0,
            "ground_truth_singletons": gt_singletons,
            "ground_truth_multi_matches": gt_multi,
        })

    return summary
