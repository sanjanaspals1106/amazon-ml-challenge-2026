"""Grouped (by S1 entity) train/validation split at ENTITY level."""

from typing import Iterable, Set, Tuple

import numpy as np


def grouped_split(s1_ids: Iterable[str], val_fraction: float = 0.2, seed: int = 42) -> Tuple[Set[str], Set[str]]:
    """
    Deterministic split of S1 entity IDs into (train_ids, val_ids). Splitting IDs (not candidate rows) guarantees no S1
    entity contributes rows to both sides, and S1 entities that received zero candidates are still assigned to a side,
    so validation metrics see them (P3's row-level `split_grouped_by_s1` cannot, because it only sees entities that
    have candidate rows).
    """
    ids = np.array(sorted(set(s1_ids)))
    if len(ids) < 2:
        raise ValueError("Need at least 2 S1 entities to make a train/validation split")
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0,1), got {val_fraction}")
    perm = np.random.default_rng(seed).permutation(len(ids))
    n_val = max(1, int(round(len(ids) * val_fraction)))
    val = set(ids[perm[:n_val]])
    train = set(ids[perm[n_val:]])
    assert not (train & val)
    return train, val
