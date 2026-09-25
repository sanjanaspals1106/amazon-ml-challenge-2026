# Candidate Generator (Blocking Pipeline) — Person 2 / Teammate A

## Overview

This module implements the candidate generation and blocking stage for the **Amazon ML Challenge 2026** (Entity Resolution across Source 1, Source 2, and Source 3).

Its role is to reduce the combinatorial search space from $\sim 10^{13}$ pairwise comparisons down to top-$K$ high-confidence candidates ($K=20\text{--}50$) per Source 1 entity, achieving $>90\text{--}95\%$ true-match recall while maintaining strict memory and formatting compliance.

---

## 1. Candidate Output Schema

When `generate_candidates()` is called, it returns a `pandas.DataFrame` with the exact schema needed for downstream pairwise feature extraction and model scoring:

| Column Name | Type | Description | Example |
| :--- | :--- | :--- | :--- |
| `s1_id` | `str` | Source 1 entity unique ID | `S1-00001` |
| `source_record_id` | `str` | Candidate entity ID from Source 2 or Source 3 | `S2-00047` |
| `source` | `str` | Origin source identifier (`'S2'` or `'S3'`) | `S2` |
| `country` | `str` | Country identifier (`'US'`, `'India'`, `'France'`) | `US` |
| `retrieval_rank` | `int` | Rank of the candidate for this S1 entity (1-indexed) | `1` |
| `retrieval_score` | `float` | Multi-channel similarity / TF-IDF overlap score | `100.0` or `18.42` |
| `retrieval_method` | `str` | Retrieval channel tag (`exact_name`, `exact_addr`, `exact_both`, `sparse_tfidf`) | `exact_name` |

---

## 2. Multi-Channel Blocking Architecture

1. **Country Partitioning:** Partitioning by `country` (100% true-match consistency on training ground truth; handles unseen test countries like France).
2. **Exact Channel Tier:** Instant exact match on normalized & token-sorted names and addresses (captures ~47% of true matches with high precision).
3. **Address TF-IDF Channel:** Inverted index over normalized address tokens with IDF weighting and high-frequency document capping ($df \le 20,000$).
4. **Name TF-IDF Channel:** Inverted index over normalized name tokens combined with 4-character token prefixes (resilient to suffix swaps, spelling variants, and typos).
5. **Score Fusion & Top-$K$ Ranking:** Summed sparse dot products combined with exact match boosts, deduplicated and sorted by score descending.

---

## 3. Python API Usage

```python
import pandas as pd
from src.blocking import CandidateGenerator

# 1. Load source data
s1_df = pd.read_csv("dataset/test/test_source1.tsv", sep="\t", dtype=str)
s2_df = pd.read_csv("dataset/test/test_source2.tsv", sep="\t", dtype=str)
s3_df = pd.read_csv("dataset/test/test_source3.tsv", sep="\t", dtype=str)

# 2. Initialize generator
generator = CandidateGenerator(
    top_k=50,             # Keep top-50 candidates per S1 entity
    df_max=20000,         # Cutoff frequent noisy tokens
    batch_size=500,       # Query batch size for sparse matrix multiplication
    verbose=True,
)

# 3. Fit corpus indices on Source 2 and Source 3
generator.fit(source2_df=s2_df, source3_df=s3_df)

# 4. Generate candidates for Source 1
candidates_df = generator.generate_candidates(source1_df=s1_df)

# candidates_df contains columns:
# ['s1_id', 'source_record_id', 'source', 'country', 'retrieval_rank', 'retrieval_score', 'retrieval_method']

# 5. Export to official competition candidate format
generator.export_candidate_pairs_tsv(
    candidates_df=candidates_df,
    all_s1_ids=s1_df["entity_id"].values,
    output_path="output/candidate_pairs.tsv",
)
```

---

## 4. CLI Execution

Run candidate generation directly from the command line:

```bash
python -m src.blocking.run_blocking \
    --source1 dataset/test/test_source1.tsv \
    --source2 dataset/test/test_source2.tsv \
    --source3 dataset/test/test_source3.tsv \
    --output-candidates output/candidates_detailed.parquet \
    --output-tsv output/candidate_pairs.tsv \
    --top-k 50
```

---

## 5. Submission Validation

Verify generated files against the official challenge validator:

```bash
python student_resource/utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```
