"""
Thin integration wrapper around P2's CandidateGenerator (src/blocking, NOT modified).

Exposes the plan's contract:

    generate_candidates(s1_df, source_df, config) -> candidate_df

where `source_df` is {"S2": s2_df, "S3": s3_df} (P2's generator indexes the two sources jointly).
Adds only orchestration: optional S1-chunked / forked-parallel generation and an on-disk cache
(keyed by the S1 set, data files and blocking parameters) so development re-runs don't repeat retrieval.

Output schema is exactly P2's:
    s1_id, source_record_id, source, country, retrieval_rank, retrieval_score, retrieval_method
"""

import hashlib
import inspect
import json
import logging
import math
import multiprocessing as mp
import time
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

from src.blocking import CandidateGenerator
from src.data_loading import CANDIDATE_COLUMNS, DataError, resolve_path

logger = logging.getLogger(__name__)

_SHARED: Dict[str, object] = {}  # generator / S1 frame shared with forked workers


def _chunk_worker(bounds: Tuple[int, int]) -> pd.DataFrame:
    start, end = bounds
    return _SHARED["gen"].generate_candidates(source1_df=_SHARED["s1"].iloc[start:end])


def generate_candidates(s1_df: pd.DataFrame, source_df: Dict[str, pd.DataFrame], config: dict) -> pd.DataFrame:
    """Fit P2's generator on S2+S3 and generate top-K candidates for every S1 row in s1_df."""
    if not isinstance(source_df, dict) or not {"S2", "S3"} <= set(source_df):
        raise DataError("generate_candidates expects source_df = {'S2': df, 'S3': df}")
    c = config.get("candidates", {})
    gen = CandidateGenerator(
        top_k=int(c.get("top_k", 50)),
        df_max=int(c.get("df_max", 20000)),
        batch_size=int(c.get("batch_size", 500)),
        prefix_weight=float(c.get("prefix_weight", 0.5)),
        verbose=True,
    )
    gen.fit(source2_df=source_df["S2"], source3_df=source_df["S3"])

    workers = int(c.get("workers", 1))
    chunk = int(c.get("chunk_size", 0))
    if workers <= 1 and chunk <= 0:
        cand = gen.generate_candidates(source1_df=s1_df)
    else:
        if chunk <= 0:
            chunk = max(200, math.ceil(len(s1_df) / (workers * 4)))
        bounds = [(i, min(i + chunk, len(s1_df))) for i in range(0, len(s1_df), chunk)]
        _SHARED["gen"], _SHARED["s1"] = gen, s1_df.reset_index(drop=True)
        gen.verbose = False
        logger.info("Candidate generation: %d chunks of <=%d S1, %d worker(s)", len(bounds), chunk, workers)
        try:
            if workers > 1:
                with mp.get_context("fork").Pool(workers) as pool:
                    parts = []
                    for k, part in enumerate(pool.imap(_chunk_worker, bounds), 1):
                        parts.append(part)
                        logger.info("  chunk %d/%d done", k, len(bounds))
            else:
                parts = [_chunk_worker(b) for b in bounds]
        finally:
            _SHARED.clear()
        cand = pd.concat(parts, ignore_index=True)
    return cand[CANDIDATE_COLUMNS]


def _file_stamp(path: Path) -> str:
    st = path.stat()
    return f"{path.name}:{st.st_size}"


def candidate_cache_key(s1_df: pd.DataFrame, config: dict, split: str, data_files, pool_desc: dict) -> str:
    """Stable key over: split, S1 ID set, blocking params, pool reduction, and input file sizes."""
    c = config.get("candidates", {})
    payload = {
        "split": split,
        "s1": hashlib.sha1("\n".join(s1_df["entity_id"].tolist()).encode()).hexdigest(),
        "top_k": c.get("top_k", 50), "df_max": c.get("df_max", 20000),
        "prefix_weight": c.get("prefix_weight", 0.5),
        "pool": pool_desc,
        "files": [_file_stamp(Path(p)) for p in data_files],
        "generator_src": hashlib.sha1(Path(inspect.getsourcefile(CandidateGenerator)).read_bytes()).hexdigest(),
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


def load_or_generate_candidates(
    s1_df: pd.DataFrame, source_df: Dict[str, pd.DataFrame], config: dict,
    split: str, data_files, pool_desc: dict, use_cache: bool = True,
) -> Tuple[pd.DataFrame, dict]:
    """Return (candidate_df, info) using the on-disk cache when available."""
    cache_dir = resolve_path(config["paths"].get("cache_dir", "outputs/cache"))
    key = candidate_cache_key(s1_df, config, split, data_files, pool_desc)
    path = cache_dir / f"candidates_{key}.parquet"
    t0 = time.time()
    if use_cache and path.exists():
        cand = pd.read_parquet(path)
        logger.info("Candidate cache HIT: %s (%s pairs)", path.name, f"{len(cand):,}")
        return cand, {"cache_hit": True, "cache_key": key, "seconds": round(time.time() - t0, 1)}
    cand = generate_candidates(s1_df, source_df, config)
    if use_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cand.to_parquet(path, index=False)
        logger.info("Candidate cache written: %s", path)
    return cand, {"cache_hit": False, "cache_key": key, "seconds": round(time.time() - t0, 1)}
