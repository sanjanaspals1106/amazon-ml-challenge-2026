"""
CLI entry point to run Candidate Generation / Blocking.
Role: Person 2 (Teammate A).

Usage:
    python -m src.blocking.run_blocking \\
        --source1 dataset/test/test_source1.tsv \\
        --source2 dataset/test/test_source2.tsv \\
        --source3 dataset/test/test_source3.tsv \\
        --output-candidates output/candidates_detailed.parquet \\
        --output-tsv output/candidate_pairs.tsv \\
        --top-k 50
"""

import argparse
import csv
import os
import sys
import time
import pandas as pd

from src.blocking.candidate_generator import CandidateGenerator


def load_tsv(path: str) -> pd.DataFrame:
    """Read a tab-separated file safely with string dtypes and QUOTE_NONE."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Source file not found: {path}")
    print(f"Loading {path} ...", flush=True)
    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        quoting=csv.QUOTE_NONE,
        encoding="utf-8",
    )
    print(f"  Loaded {len(df):,} rows from {os.path.basename(path)}")
    return df


# Globals shared with forked worker processes (chunked / parallel mode)
_GEN = None
_S1 = None


def _run_chunk(bounds):
    """Worker: generate candidates for S1 rows [start, end) using the already-fitted (forked) generator."""
    start, end = bounds
    return start, _GEN.generate_candidates(source1_df=_S1.iloc[start:end])


def run_chunked(generator, s1_df, args):
    """
    Memory-safe / parallel driver: process S1 in chunks (optionally in several forked workers), stream each
    chunk's results to disk in S1 order. Uses generator.generate_candidates() and
    generator.export_candidate_pairs_tsv() unchanged, so per-entity results and schema are identical.
    Returns (total_rows, n_s1_with_candidates, first_chunk_head).
    """
    import multiprocessing as mp
    import shutil

    global _GEN, _S1
    _GEN, _S1 = generator, s1_df
    chunk = args.chunk_size if args.chunk_size > 0 else 10000
    bounds = [(i, min(i + chunk, len(s1_df))) for i in range(0, len(s1_df), chunk)]
    all_ids = s1_df["entity_id"].values
    parquet_out = args.output_candidates.endswith(".parquet")
    writer = None
    total_rows, n_with, head = 0, 0, None
    t0 = time.time()
    part_path = (args.output_tsv or "") + ".part"
    if args.output_tsv:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_tsv)), exist_ok=True)
        with open(args.output_tsv, "w", encoding="utf-8") as f:
            f.write("source1_entity_id\tcandidate_entity_ids\n")
    os.makedirs(os.path.dirname(os.path.abspath(args.output_candidates)), exist_ok=True)
    if not parquet_out:
        open(args.output_candidates, "w").close()

    generator.verbose = False  # silence per-chunk logs from workers
    print(f"Chunked mode: {len(bounds)} chunks of <= {chunk:,} S1 entities, {args.workers} worker(s)", flush=True)
    pool = mp.get_context("fork").Pool(args.workers) if args.workers > 1 else None
    try:
        it = pool.imap(_run_chunk, bounds) if pool else map(_run_chunk, bounds)
        for n_done, (start, cand) in enumerate(it, start=1):
            end = min(start + chunk, len(s1_df))
            if head is None:
                head = cand.head(5)
            total_rows += len(cand)
            n_with += cand["s1_id"].nunique() if len(cand) else 0
            if len(cand):
                if parquet_out:
                    import pyarrow as pa
                    import pyarrow.parquet as pq
                    table = pa.Table.from_pandas(cand, preserve_index=False)
                    if writer is None:
                        writer = pq.ParquetWriter(args.output_candidates, table.schema)
                    writer.write_table(table)
                else:
                    cand.to_csv(args.output_candidates, sep="\t", index=False, mode="a", header=(n_done == 1))
            if args.output_tsv:
                generator.export_candidate_pairs_tsv(cand, all_ids[start:end], part_path)
                with open(part_path, "r", encoding="utf-8") as src, open(args.output_tsv, "a", encoding="utf-8") as dst:
                    next(src)  # skip per-chunk header
                    shutil.copyfileobj(src, dst)
                os.remove(part_path)
            el = time.time() - t0
            eta = el / n_done * (len(bounds) - n_done)
            print(f"  chunk {n_done}/{len(bounds)} done | {total_rows:,} pairs | elapsed {el/60:.1f} min | ETA {eta/60:.1f} min", flush=True)
    finally:
        if pool:
            pool.close()
            pool.join()
        if writer is not None:
            writer.close()
        generator.verbose = True
    return total_rows, n_with, head


def main():
    parser = argparse.ArgumentParser(description="Candidate Generator for ML Challenge 2026 (Person 2 / Teammate A)")
    parser.add_argument("--source1", "-s1", required=True, help="Path to Source 1 TSV file")
    parser.add_argument("--source2", "-s2", required=True, help="Path to Source 2 TSV file")
    parser.add_argument("--source3", "-s3", required=True, help="Path to Source 3 TSV file")
    parser.add_argument("--output-candidates", "-o", default="output/candidates_detailed.parquet",
                        help="Path to save detailed candidate DataFrame (parquet or csv/tsv)")
    parser.add_argument("--output-tsv", "-t", default="output/candidate_pairs.tsv",
                        help="Path to save submission-format candidate_pairs.tsv")
    parser.add_argument("--top-k", "-k", type=int, default=50, help="Maximum candidates per S1 entity")
    parser.add_argument("--df-max", type=int, default=20000, help="Max document frequency for common token cutoff")
    parser.add_argument("--batch-size", type=int, default=500, help="Query batch size for sparse retrieval")
    parser.add_argument("--chunk-size", type=int, default=0,
                        help="Process S1 in chunks of this many entities and stream results to disk (0 = single in-memory call)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel worker processes for chunked mode (fork; implies chunked mode if > 1)")
    args = parser.parse_args()

    t_start = time.time()
    print("=" * 70)
    print(" Amazon ML Challenge 2026 — Candidate Generator (Teammate A / Person 2)")
    print("=" * 70)

    # 1. Load data
    s1_df = load_tsv(args.source1)
    s2_df = load_tsv(args.source2)
    s3_df = load_tsv(args.source3)

    # 2. Initialize generator
    generator = CandidateGenerator(
        top_k=args.top_k,
        df_max=args.df_max,
        batch_size=args.batch_size,
        verbose=True,
    )

    # 3. Fit on S2 + S3
    generator.fit(source2_df=s2_df, source3_df=s3_df)

    if args.workers > 1 or args.chunk_size > 0:
        # 4-6. Chunked / parallel generation with streamed output
        total_rows, n_with, head = run_chunked(generator, s1_df, args)
        print("\nCandidate Generator Output Sample (Top 5 rows):")
        print(head.to_string(index=False))
        print(f"\nSaved detailed candidates to: {args.output_candidates}")
        if args.output_tsv:
            print(f"Saved submission candidate file to: {args.output_tsv}")
        print(f"Total candidate pairs: {total_rows:,} for {n_with:,}/{len(s1_df):,} S1 entities")
    else:
        # 4. Generate candidates for S1
        candidates_df = generator.generate_candidates(source1_df=s1_df)

        # Display sample output
        print("\nCandidate Generator Output Sample (Top 5 rows):")
        print(candidates_df.head(5).to_string(index=False))

        # 5. Save detailed candidates
        os.makedirs(os.path.dirname(os.path.abspath(args.output_candidates)), exist_ok=True)
        if args.output_candidates.endswith(".parquet"):
            candidates_df.to_parquet(args.output_candidates, index=False)
        else:
            candidates_df.to_csv(args.output_candidates, sep="\t", index=False)
        print(f"\nSaved detailed candidates to: {args.output_candidates}")

        # 6. Save competition candidate_pairs.tsv
        if args.output_tsv:
            all_s1_ids = s1_df["entity_id"].values
            tsv_path = generator.export_candidate_pairs_tsv(
                candidates_df=candidates_df,
                all_s1_ids=all_s1_ids,
                output_path=args.output_tsv,
            )
            print(f"Saved submission candidate file to: {tsv_path}")


    print(f"\nCandidate generation complete in {time.time() - t_start:.2f}s.")


if __name__ == "__main__":
    main()
