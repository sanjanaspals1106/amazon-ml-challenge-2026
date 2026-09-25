"""
Data-loading orchestration for the baseline pipeline (P1).

Responsibilities: resolve config paths (relative to the repo root), read the TSVs safely, validate
schemas / IDs, sample S1 entities deterministically, restrict the S2/S3 pool for wiring runs, and
validate candidate frames coming out of the blocking stage.

Ground-truth lookups and label building are NOT implemented here: they are P3's
(`src.model.labels.build_ground_truth_lookup` / `build_labels`).
"""

import csv
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
GROUND_TRUTH_COLUMNS = ["source1_entity_id", "matched_entity_ids"]
CANDIDATE_COLUMNS = [
    "s1_id", "source_record_id", "source", "country",
    "retrieval_rank", "retrieval_score", "retrieval_method",
]
VALID_SOURCES = ("S2", "S3")


class DataError(ValueError):
    """Raised for invalid / inconsistent input data with an actionable message."""


def resolve_path(path: str) -> Path:
    """Resolve a config path: absolute paths are kept, relative ones are relative to the repo root."""
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def dataset_paths(config: dict, split: str) -> Dict[str, Path]:
    """Return {source1, source2, source3[, ground_truth]} paths for split in {'train','test'}."""
    try:
        root = resolve_path(config["paths"]["dataset_root"])
        files = config["paths"][split]
    except KeyError as e:
        raise DataError(f"Config is missing paths.{split} / paths.dataset_root (missing key: {e})") from e
    return {name: root / rel for name, rel in files.items()}


def load_tsv(path: Path, required_columns: List[str], name: str) -> pd.DataFrame:
    """Read a TSV exactly as recommended for this dataset (see README): strings only, no NA parsing, no quoting."""
    if not Path(path).exists():
        raise FileNotFoundError(f"{name} file not found: {path} (check paths.* in the config)")
    df = pd.read_csv(
        path, sep="\t", dtype=str, keep_default_na=False, na_filter=False,
        quoting=csv.QUOTE_NONE, encoding="utf-8",
    )
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise DataError(f"{name} ({path}) is missing required columns {missing}; found {list(df.columns)}")
    return df


def validate_source_frame(df: pd.DataFrame, prefix: str, name: str) -> None:
    """Check IDs (non-empty, unique, right prefix) and that country is present."""
    ids = df["entity_id"]
    if (ids.str.strip() == "").any():
        raise DataError(f"{name}: {(ids.str.strip() == '').sum()} null/empty entity_id values")
    if ids.duplicated().any():
        raise DataError(f"{name}: {int(ids.duplicated().sum())} duplicate entity_id values")
    bad = ~ids.str.startswith(prefix + "-")
    if bad.any():
        raise DataError(f"{name}: {int(bad.sum())} IDs do not start with '{prefix}-' (e.g. {ids[bad].iloc[0]!r})")
    if (df["country"].str.strip() == "").any():
        raise DataError(f"{name}: {(df['country'].str.strip() == '').sum()} rows have an empty country")


def load_split(config: dict, split: str) -> Dict[str, pd.DataFrame]:
    """Load S1/S2/S3 (and ground truth for train) for a split. Returns dict with keys s1, s2, s3[, gt]."""
    paths = dataset_paths(config, split)
    for key in ("source1", "source2", "source3"):
        if key not in paths:
            raise DataError(f"Config paths.{split} must define '{key}'")
    data = {
        "s1": load_tsv(paths["source1"], SOURCE_COLUMNS, f"{split} source1"),
        "s2": load_tsv(paths["source2"], SOURCE_COLUMNS, f"{split} source2"),
        "s3": load_tsv(paths["source3"], SOURCE_COLUMNS, f"{split} source3"),
    }
    for key, prefix in (("s1", "S1"), ("s2", "S2"), ("s3", "S3")):
        validate_source_frame(data[key], prefix, f"{split} {key}")
    if split == "train":
        if "ground_truth" not in paths:
            raise DataError("Config paths.train must define 'ground_truth'")
        data["gt"] = load_tsv(paths["ground_truth"], GROUND_TRUTH_COLUMNS, "train ground truth")
        gt = data["gt"]
        if gt["source1_entity_id"].duplicated().any():
            raise DataError("train ground truth has duplicate source1_entity_id rows")
        unknown = ~gt["source1_entity_id"].isin(data["s1"]["entity_id"])
        if unknown.any():
            raise DataError(f"train ground truth references {int(unknown.sum())} S1 IDs that are not in train source1")
    logger.info("Loaded %s: S1=%s S2=%s S3=%s", split, f"{len(data['s1']):,}", f"{len(data['s2']):,}", f"{len(data['s3']):,}")
    return data


def sample_s1(s1_df: pd.DataFrame, n: Optional[int], seed: int) -> pd.DataFrame:
    """Deterministic random sample of S1 entities (sorted by original order for stability). n=None -> all."""
    if n is None or n <= 0 or n >= len(s1_df):
        return s1_df.reset_index(drop=True)
    idx = np.sort(np.random.default_rng(seed).choice(len(s1_df), size=n, replace=False))
    return s1_df.iloc[idx].reset_index(drop=True)


def restrict_pool(
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    frac: float,
    seed: int,
    keep_ids: Optional[Set[str]] = None,
):
    """
    WIRING-ONLY pool reduction. Keeps every record in keep_ids (e.g. true matches of the sampled S1) plus a random
    `frac` of all other records. frac >= 1 returns the pools unchanged. A reduced pool has far fewer look-alike
    records than the real one, so metrics from it are NOT representative.
    """
    if frac >= 1.0:
        return s2_df, s3_df
    rng = np.random.default_rng(seed)
    out = []
    for df in (s2_df, s3_df):
        keep = rng.random(len(df)) < frac
        if keep_ids:
            keep |= df["entity_id"].isin(keep_ids).values
        out.append(df[keep].reset_index(drop=True))
    logger.warning("pool_sample_frac=%.3f: S2 pool %s -> %s, S3 pool %s -> %s (metrics NOT representative)",
                   frac, f"{len(s2_df):,}", f"{len(out[0]):,}", f"{len(s3_df):,}", f"{len(out[1]):,}")
    return out[0], out[1]


def validate_candidates(cand: pd.DataFrame, expected_s1_ids: Optional[Iterable[str]] = None) -> Dict[str, int]:
    """
    Validate the blocking output. Raises DataError on: missing columns, empty set, null IDs, invalid source values,
    source/ID-prefix mismatch, duplicate (s1_id, source_record_id) pairs, S1 IDs outside the expected set.
    Returns small diagnostics (e.g. number of S1 entities without any candidate).
    """
    missing = [c for c in CANDIDATE_COLUMNS if c not in cand.columns]
    if missing:
        raise DataError(f"Candidate frame is missing columns {missing}; found {list(cand.columns)}")
    if len(cand) == 0:
        raise DataError("Candidate set is empty: blocking returned no candidates (check country labels / pool)")
    for c in ("s1_id", "source_record_id"):
        if cand[c].isna().any() or (cand[c].astype(str).str.strip() == "").any():
            raise DataError(f"Candidate frame has null/empty values in '{c}'")
    bad_src = ~cand["source"].isin(VALID_SOURCES)
    if bad_src.any():
        raise DataError(f"Invalid source values {sorted(cand.loc[bad_src, 'source'].unique())[:5]}; expected {VALID_SOURCES}")
    mismatch = cand["source_record_id"].str[:2] != cand["source"]
    if mismatch.any():
        raise DataError(f"{int(mismatch.sum())} candidates whose source_record_id prefix disagrees with 'source'")
    dups = cand.duplicated(["s1_id", "source_record_id"])
    if dups.any():
        raise DataError(f"{int(dups.sum())} duplicate (s1_id, source_record_id) candidate pairs")
    info = {"n_pairs": int(len(cand)), "n_s1_with_candidates": int(cand["s1_id"].nunique())}
    if expected_s1_ids is not None:
        expected = set(expected_s1_ids)
        extra = set(cand["s1_id"].unique()) - expected
        if extra:
            raise DataError(f"{len(extra)} candidate S1 IDs are not in the evaluated S1 set (e.g. {sorted(extra)[:3]})")
        info["n_s1_without_candidates"] = int(len(expected) - info["n_s1_with_candidates"])
    return info
