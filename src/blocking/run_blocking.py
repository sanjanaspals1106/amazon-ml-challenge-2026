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
    parser.add_argument("--exact-name-cap", type=int, default=8,
                        help="Max exact_name-tier candidates kept per S1 (reduces crowding out other tiers)")
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
        exact_name_cap=args.exact_name_cap,
        verbose=True,
    )

    # 3. Fit on S2 + S3
    generator.fit(source2_df=s2_df, source3_df=s3_df)

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
