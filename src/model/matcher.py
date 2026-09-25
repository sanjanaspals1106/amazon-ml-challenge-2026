"""
Gradient-boosted tree pair matcher model (M2).

Uses XGBoost (XGBClassifier) by default, with automatic fallback to
scikit-learn's HistGradientBoostingClassifier if XGBoost is unavailable.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Non-feature metadata columns to exclude from training/inference matrices
EXCLUDED_COLUMNS = {
    "s1_id",
    "source_record_id",
    "source",
    "label",
    "target",
    "kind",
    "retrieval_method",
    "country",
}


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Return all numerical feature column names, excluding metadata and identity columns."""
    cols = []
    for c in df.columns:
        if c not in EXCLUDED_COLUMNS:
            # Check if column is numeric or can be safely cast
            if pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c]):
                cols.append(c)
    return sorted(cols)


class EntityResolutionMatcher:
    """
    Supervised gradient-boosted pairwise classifier.
    Predicts the probability that a candidate record matches an S1 entity.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.model_type = self.config.get("model_type", "xgboost").lower()
        self.feature_names_: List[str] = []
        self.model_: Any = None
        self._init_model()

    def _init_model(self) -> None:
        """Instantiate the underlying classifier based on configuration."""
        params = self.config.get("model_params", {})

        if self.model_type == "xgboost":
            try:
                import xgboost as xgb
                # Sensible defaults for entity resolution ranking/classification
                xgb_params = {
                    "n_estimators": params.get("n_estimators", 250),
                    "max_depth": params.get("max_depth", 6),
                    "learning_rate": params.get("learning_rate", 0.08),
                    "subsample": params.get("subsample", 0.8),
                    "colsample_bytree": params.get("colsample_bytree", 0.8),
                    "eval_metric": params.get("eval_metric", "logloss"),
                    "random_state": params.get("random_state", 42),
                    "n_jobs": params.get("n_jobs", -1),
                    "tree_method": params.get("tree_method", "auto"),
                    "min_child_weight": params.get("min_child_weight", 1),
                }
                if "scale_pos_weight" in params:
                    xgb_params["scale_pos_weight"] = params["scale_pos_weight"]

                self.model_ = xgb.XGBClassifier(**xgb_params)
                logger.info(f"Initialized XGBClassifier with params: {xgb_params}")
            except ImportError as e:
                logger.warning(f"XGBoost not available ({e}). Falling back to HistGradientBoostingClassifier.")
                self.model_type = "hist_gradient_boosting"

        if self.model_type != "xgboost":
            from sklearn.ensemble import HistGradientBoostingClassifier
            hgb_params = {
                "max_iter": params.get("n_estimators", 200),
                "max_depth": params.get("max_depth", 8),
                "learning_rate": params.get("learning_rate", 0.08),
                "random_state": params.get("random_state", 42),
            }
            if "class_weight" in params:
                hgb_params["class_weight"] = params["class_weight"]

            self.model_ = HistGradientBoostingClassifier(**hgb_params)
            logger.info(f"Initialized HistGradientBoostingClassifier with params: {hgb_params}")

    def prepare_features(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Validate, align, and extract numeric feature matrix.
        """
        if isinstance(X, pd.DataFrame):
            if not self.feature_names_:
                # First time seeing features (at fit time)
                self.feature_names_ = get_feature_columns(X)
            # Align columns strictly to feature_names_
            missing = [c for c in self.feature_names_ if c not in X.columns]
            if missing:
                raise ValueError(f"Input features missing required columns: {missing}")
            sub = X[self.feature_names_].astype(np.float32)
            return sub.values
        else:
            return np.asarray(X, dtype=np.float32)

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        sample_weight: Optional[np.ndarray] = None
    ) -> "EntityResolutionMatcher":
        """
        Fit the gradient-boosted tree matcher.

        Args:
            X: Feature matrix or DataFrame with features.
            y: Binary target labels (1 for match, 0 for non-match).
            sample_weight: Optional per-sample weights.
        """
        X_mat = self.prepare_features(X)
        y_arr = np.asarray(y, dtype=np.int32)

        # Check positive rate
        n_pos = int((y_arr == 1).sum())
        n_neg = int((y_arr == 0).sum())
        logger.info(f"Fitting {self.model_type} matcher on {len(y_arr)} pairs ({n_pos} positive, {n_neg} negative).")

        # Auto-compute scale_pos_weight for XGBoost if requested in config and not already set
        if self.model_type == "xgboost" and self.config.get("auto_scale_pos_weight", False):
            if n_pos > 0 and n_neg > 0:
                scale_val = float(n_neg / n_pos)
                self.model_.set_params(scale_pos_weight=scale_val)
                logger.info(f"Automatically set scale_pos_weight to {scale_val:.2f}")

        if sample_weight is not None:
            self.model_.fit(X_mat, y_arr, sample_weight=sample_weight)
        else:
            self.model_.fit(X_mat, y_arr)

        return self

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict match probability / confidence for each pair.

        Returns:
            1D numpy array of probabilities for class 1 (match).
        """
        X_mat = self.prepare_features(X)
        if len(X_mat) == 0:
            return np.array([], dtype=np.float32)
        probs = self.model_.predict_proba(X_mat)[:, 1]
        return probs.astype(np.float32)

    def get_feature_importances(self) -> Optional[pd.DataFrame]:
        """Return a sorted DataFrame of feature importances if supported."""
        if not self.feature_names_:
            return None
        if hasattr(self.model_, "feature_importances_"):
            importances = self.model_.feature_importances_
            df = pd.DataFrame({
                "feature": self.feature_names_,
                "importance": importances
            }).sort_values("importance", ascending=False).reset_index(drop=True)
            return df
        return None

    def save(self, filepath: str) -> None:
        """Save trained model and feature schema to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        bundle = {
            "model": self.model_,
            "model_type": self.model_type,
            "feature_names": self.feature_names_,
            "config": self.config,
        }
        joblib.dump(bundle, filepath)
        logger.info(f"Saved matcher model to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "EntityResolutionMatcher":
        """Load trained model and feature schema from disk."""
        bundle = joblib.load(filepath)
        matcher = cls(config=bundle.get("config", {}))
        matcher.model_ = bundle["model"]
        matcher.model_type = bundle["model_type"]
        matcher.feature_names_ = bundle["feature_names"]
        logger.info(f"Loaded {matcher.model_type} matcher from {filepath} with {len(matcher.feature_names_)} features.")
        return matcher
