"""
Inference and scoring module for M2 matching model.

Contract:
    predict_match_scores(model, feature_df) -> scores
"""

import logging
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd

from src.model.matcher import EntityResolutionMatcher

logger = logging.getLogger(__name__)


def predict_match_scores(
    model: EntityResolutionMatcher,
    feature_df: pd.DataFrame
) -> np.ndarray:
    """
    Predict match probabilities for candidate pairs.

    Args:
        model: Trained EntityResolutionMatcher instance.
        feature_df: DataFrame containing candidate identity and pair features.

    Returns:
        1D numpy array of float32 probabilities indicating match confidence.
    """
    if feature_df is None or len(feature_df) == 0:
        return np.array([], dtype=np.float32)

    return model.predict_proba(feature_df)


def score_candidates(
    model: EntityResolutionMatcher,
    feature_df: pd.DataFrame,
    score_col: str = "match_score"
) -> pd.DataFrame:
    """
    Attach match probability scores directly to feature_df / candidate_df.

    Returns:
        DataFrame with score_col added.
    """
    df = feature_df.copy()
    scores = predict_match_scores(model, df)
    df[score_col] = scores
    return df
