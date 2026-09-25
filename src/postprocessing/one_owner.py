"""
One-owner post-processing (team plan section 11).

Ground truth assigns every S2/S3 record to AT MOST ONE S1 entity. After thresholding, several S1 entities may
claim the same source record; this keeps only the strongest claim:

- One S1 may keep many source records; zero matches are allowed (nothing is forced).
- For a record claimed by >1 S1: the highest-scoring S1 wins.
- Near-tie handling: if the best score beats the runner-up by no more than `near_tie_margin`, confidence is
  insufficient and the record is dropped for every claimant (set margin=None to disable and always keep the best,
  breaking exact ties by smallest S1 id).
"""

from typing import Dict, Optional, Tuple

import pandas as pd


def resolve_one_owner(
    scored: pd.DataFrame,
    threshold: float,
    near_tie_margin: Optional[float] = 0.05,
    score_col: str = "score",
    s1_col: str = "s1_id",
    record_col: str = "source_record_id",
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Args:
        scored: candidate pairs with s1_col, record_col and score_col (any extra columns are preserved).
        threshold: pairs with score < threshold are rejected first.
    Returns:
        (accepted_pairs, info) where accepted_pairs has at most one row per source record.
    """
    acc = scored[scored[score_col] >= threshold]
    n_thresholded = len(acc)
    if n_thresholded == 0:
        return acc.copy(), {"n_thresholded": 0, "n_contested_records": 0, "n_near_tie_dropped_records": 0, "n_pairs_removed": 0}

    acc = acc.sort_values([record_col, score_col, s1_col], ascending=[True, False, True], kind="mergesort")
    rank = acc.groupby(record_col, sort=False).cumcount()
    top = acc[rank == 0].set_index(record_col, drop=False)
    second = acc[rank == 1].set_index(record_col)[score_col]

    gap = top[score_col] - second.reindex(top.index)  # NaN where uncontested
    contested = gap.notna()
    if near_tie_margin is None:
        drop = pd.Series(False, index=top.index)
    else:
        drop = gap.le(near_tie_margin).fillna(False)

    kept = top[~drop.values].reset_index(drop=True)
    info = {
        "n_thresholded": int(n_thresholded),
        "n_contested_records": int(contested.sum()),
        "n_near_tie_dropped_records": int(drop.sum()),
        "n_pairs_removed": int(n_thresholded - len(kept)),
    }
    return kept, info


def verify_one_owner(matches: pd.DataFrame, record_col: str = "source_record_id") -> None:
    """Raise AssertionError if any source record is owned by more than one S1 entity."""
    dup = matches[record_col].duplicated()
    if dup.any():
        raise AssertionError(f"One-owner constraint violated for {int(dup.sum())} source records "
                             f"(e.g. {matches.loc[dup, record_col].iloc[0]})")
