"""Threshold grid + selection on grouped validation data (team plan section 12)."""

from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.postprocessing.metrics import evaluate_entities
from src.postprocessing.one_owner import resolve_one_owner


def make_threshold_grid(cfg: Optional[dict]) -> List[float]:
    """Grid from config: {'values': [...]} or {'start','stop','step'} (inclusive). Default 0.30..0.90 step 0.05."""
    cfg = cfg or {}
    if cfg.get("values"):
        grid = [float(v) for v in cfg["values"]]
    else:
        start, stop, step = float(cfg.get("start", 0.30)), float(cfg.get("stop", 0.90)), float(cfg.get("step", 0.05))
        if step <= 0 or stop < start:
            raise ValueError(f"Invalid threshold grid start={start} stop={stop} step={step}")
        n = int(round((stop - start) / step)) + 1
        grid = [round(start + i * step, 4) for i in range(n)]
    if not grid:
        raise ValueError("Threshold grid is empty")
    return sorted(set(grid))


def apply_decision_rule(
    scored: pd.DataFrame, threshold: float, one_owner_cfg: Optional[dict], score_col: str = "score",
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Candidate-level threshold, then (optionally) the one-owner rule. Returns (accepted_pairs, one_owner_info)."""
    if one_owner_cfg is not None and one_owner_cfg.get("enabled", True):
        return resolve_one_owner(scored, threshold, near_tie_margin=one_owner_cfg.get("near_tie_margin", 0.05), score_col=score_col)
    return scored[scored[score_col] >= threshold].copy(), {}


def search_threshold(
    scored: pd.DataFrame,
    s1_ids: Iterable[str],
    truth: pd.DataFrame,
    grid: List[float],
    metric: str = "macro_f05",
    one_owner_cfg: Optional[dict] = None,
    score_col: str = "score",
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """
    Evaluate every threshold with the real decision rule (threshold [+ one-owner]) at ENTITY level over ALL s1_ids.
    Best = highest `metric`; ties resolve to the HIGHER threshold (precision-oriented, deterministic).
    """
    s1_ids = list(s1_ids)
    rows = []
    for t in grid:
        acc, info = apply_decision_rule(scored, t, one_owner_cfg, score_col)
        m = evaluate_entities(s1_ids, truth, acc)
        pre = evaluate_entities(s1_ids, truth, scored[scored[score_col] >= t]) if one_owner_cfg else m
        rows.append({"threshold": t, **{k: m[k] for k in (
            "macro_f05", "macro_precision", "macro_recall", "singleton_f05", "non_singleton_f05",
            "pair_precision", "pair_recall", "pair_fp", "n_predicted_pairs")},
            "macro_f05_before_one_owner": pre["macro_f05"], "n_pairs_before_one_owner": pre["n_predicted_pairs"], **info})
    table = pd.DataFrame(rows)
    if metric not in table.columns:
        raise ValueError(f"Unknown threshold metric '{metric}'; available: {list(table.columns)}")
    best_val = table[metric].max()
    best = table[np.isclose(table[metric], best_val)].sort_values("threshold").iloc[-1]
    return best.to_dict(), table
