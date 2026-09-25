"""
Entity-level (macro) validation metrics, competition-exact.

For every evaluated S1 entity (INCLUDING entities that received no candidates and singletons):
- true set empty:  F0.5 = 1 if nothing predicted else 0            (precision/recall: 1/1 or 0/1)
- true set non-empty, nothing predicted: F0.5 = 0                   (precision 1, recall 0)
- otherwise F0.5 = 1.25 P R / (0.25 P + R)
which is exactly `src.model.training.compute_entity_f05` (P3); tests assert equality. Macro metrics are averages over
entities. Segments:
  * country segments: subset of S1 entities by S1 country
  * source segments (S2 / S3): predictions AND truth restricted to that source's IDs, over all evaluated entities
Pair-level micro counts (TP/FP/FN) are computed against ALL true pairs, so they include matches blocking missed.
"""

from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd


def truth_pairs(gt_lookup: Dict[str, set], s1_ids: Iterable[str]) -> pd.DataFrame:
    """All true (s1_id, source_record_id) pairs for the given S1 entities (from P3's ground-truth lookup)."""
    rows = [(s1, rid) for s1 in s1_ids for rid in gt_lookup.get(s1, ())]
    return pd.DataFrame(rows, columns=["s1_id", "source_record_id"])


def _entity_scores(n_true: np.ndarray, n_pred: np.ndarray, tp: np.ndarray):
    f = np.zeros(len(n_true))
    p = np.zeros(len(n_true))
    r = np.zeros(len(n_true))
    single = n_true == 0
    # singletons
    f[single & (n_pred == 0)] = 1.0
    p[single] = np.where(n_pred[single] == 0, 1.0, 0.0)
    r[single] = 1.0
    # non-singletons
    ns = ~single
    none = ns & (n_pred == 0)
    p[none] = 1.0  # P3 convention: empty prediction -> precision 1, recall 0, F 0
    has = ns & (n_pred > 0)
    p[has] = tp[has] / n_pred[has]
    r[has] = tp[has] / n_true[has]
    denom = 0.25 * p[has] + r[has]
    f[has] = np.divide(1.25 * p[has] * r[has], denom, out=np.zeros(has.sum()), where=denom > 0)
    return f, p, r, single


def evaluate_entities(
    s1_ids: Iterable[str],
    truth: pd.DataFrame,
    pred: pd.DataFrame,
    source: Optional[str] = None,
) -> Dict[str, float]:
    """
    Args:
        s1_ids: all evaluated S1 entity IDs.
        truth: DataFrame(s1_id, source_record_id) of true pairs for those entities.
        pred: DataFrame(s1_id, source_record_id) of predicted (accepted) pairs.
        source: 'S2' / 'S3' to restrict truth and predictions to that source; None = both.
    """
    s1_ids = pd.Index(pd.unique(pd.Series(list(s1_ids))))
    if len(s1_ids) == 0:
        raise ValueError("evaluate_entities called with no S1 entities")
    if source is not None:
        truth = truth[truth["source_record_id"].str.startswith(source + "-")]
        pred = pred[pred["source_record_id"].str.startswith(source + "-")]
    pred = pred[pred["s1_id"].isin(s1_ids)]
    n_true = truth.groupby("s1_id").size().reindex(s1_ids, fill_value=0).to_numpy()
    n_pred = pred.groupby("s1_id").size().reindex(s1_ids, fill_value=0).to_numpy()
    hit = pred.merge(truth.drop_duplicates(), on=["s1_id", "source_record_id"], how="inner")
    tp = hit.groupby("s1_id").size().reindex(s1_ids, fill_value=0).to_numpy()

    f, p, r, single = _entity_scores(n_true.astype(float), n_pred.astype(float), tp.astype(float))
    TP, FP, FN = int(tp.sum()), int(n_pred.sum() - tp.sum()), int(n_true.sum() - tp.sum())
    pp = TP / (TP + FP) if TP + FP else 0.0
    pr = TP / (TP + FN) if TP + FN else 0.0
    pf = 1.25 * pp * pr / (0.25 * pp + pr) if (0.25 * pp + pr) > 0 else 0.0
    return {
        "macro_f05": float(f.mean()), "macro_precision": float(p.mean()), "macro_recall": float(r.mean()),
        "singleton_f05": float(f[single].mean()) if single.any() else float("nan"),
        "non_singleton_f05": float(f[~single].mean()) if (~single).any() else float("nan"),
        "n_entities": int(len(s1_ids)), "n_singletons": int(single.sum()),
        "pair_tp": TP, "pair_fp": FP, "pair_fn": FN,  # pair_fp = "false merges"
        "pair_precision": pp, "pair_recall": pr, "pair_f05": pf,
        "n_predicted_pairs": int(n_pred.sum()), "n_true_pairs": int(n_true.sum()),
    }


def evaluate_with_segments(
    s1_ids: Iterable[str],
    truth: pd.DataFrame,
    pred: pd.DataFrame,
    country_by_s1: Dict[str, str],
) -> Dict[str, object]:
    """Overall + per-country + per-source metrics."""
    s1_ids = list(pd.unique(pd.Series(list(s1_ids))))
    out: Dict[str, object] = {"overall": evaluate_entities(s1_ids, truth, pred)}
    countries = sorted({country_by_s1[s] for s in s1_ids if s in country_by_s1})
    out["country"] = {}
    for c in countries:
        sub = [s for s in s1_ids if country_by_s1.get(s) == c]
        out["country"][c] = evaluate_entities(sub, truth[truth["s1_id"].isin(sub)], pred[pred["s1_id"].isin(sub)])
    out["source"] = {src: evaluate_entities(s1_ids, truth, pred, source=src) for src in ("S2", "S3")}
    return out


def candidate_recall(truth: pd.DataFrame, candidates: pd.DataFrame) -> Dict[str, float]:
    """Share of true pairs that blocking retrieved (the ceiling for any matcher)."""
    if len(truth) == 0:
        return {"true_pairs": 0, "retrieved": 0, "candidate_recall": float("nan")}
    hit = truth.merge(candidates[["s1_id", "source_record_id"]].drop_duplicates(), on=["s1_id", "source_record_id"], how="inner")
    return {"true_pairs": int(len(truth)), "retrieved": int(len(hit)), "candidate_recall": len(hit) / len(truth)}
