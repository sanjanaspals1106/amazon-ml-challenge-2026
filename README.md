# Candidate Generator (Blocking Pipeline) — Person 2 / Teammate A

## Overview

This module implements the candidate generation and blocking stage for the **Amazon ML Challenge 2026** (Entity Resolution across Source 1, Source 2, and Source 3).

Its role is to reduce the combinatorial search space from $\sim 10^{13}$ pairwise comparisons down to top-$K$ high-confidence candidates ($K=20\text{--}50$) per Source 1 entity, targeting $>90\text{--}95\%$ true-match recall while maintaining strict memory and formatting compliance. (Recall on the full candidate pool has not yet been measured for this implementation — see [Scale, runtime and memory](#6-scale-runtime-and-memory).)

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

0. **Text Normalization** (`src/preprocessing/normalizer.py`, re-exported via `src/blocking/text_processor.py`): lowercase, `&` → `and`, Latin-diacritic folding (Indic scripts untouched), punctuation → space, abbreviation expansion (`Pvt/Ltd/Corp/Inc/Co`, and for addresses `Rd/St/Ave/…`), and removal of `NULL` / `N/A` placeholders in addresses.
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

# 1. Load source data (dataset lives under student_resource/dataset/).
#    Use keep_default_na=False and QUOTE_NONE: names such as "NA"/"null" and empty addresses are real values,
#    and a few lines contain a bare double-quote character.
import csv
read = lambda path: pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, na_filter=False, quoting=csv.QUOTE_NONE)
s1_df = read("student_resource/dataset/test/test_source1.tsv")
s2_df = read("student_resource/dataset/test/test_source2.tsv")
s3_df = read("student_resource/dataset/test/test_source3.tsv")

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

> **Note:** `generate_candidates()` keeps every result row in memory, so this in-memory API is only suitable for small/medium runs (roughly up to a few million candidate rows). For the full test set use the chunked CLI in section 4.

---

## 4. CLI Execution

Run from the repository root (the entry point is `run_blocking.py` at the top level, not a module inside `src/blocking`):

```bash
python run_blocking.py --help
```

**Small / medium run** (single process, results held in memory):

```bash
python run_blocking.py \
    --source1 student_resource/dataset/test/test_source1.tsv \
    --source2 student_resource/dataset/test/test_source2.tsv \
    --source3 student_resource/dataset/test/test_source3.tsv \
    --output-candidates output/candidates_detailed.parquet \
    --output-tsv output/candidate_pairs.tsv \
    --top-k 50
```

**Full test set** (chunked, streamed to disk, optionally parallel):

```bash
python run_blocking.py \
    --source1 student_resource/dataset/test/test_source1.tsv \
    --source2 student_resource/dataset/test/test_source2.tsv \
    --source3 student_resource/dataset/test/test_source3.tsv \
    --output-candidates output/candidates_detailed.parquet \
    --output-tsv output/candidate_pairs.tsv \
    --top-k 50 --workers 6 --chunk-size 10000
```

| Flag | Default | Meaning |
| :--- | :--- | :--- |
| `--source1/2/3` | required | Paths to the S1 / S2 / S3 TSV files |
| `--output-candidates`, `-o` | `output/candidates_detailed.parquet` | Detailed candidates (parquet, or TSV if the name does not end in `.parquet`) |
| `--output-tsv`, `-t` | `output/candidate_pairs.tsv` | Submission-format `candidate_pairs.tsv` |
| `--top-k`, `-k` | `50` | Max candidates per S1 entity |
| `--df-max` | `20000` | Document-frequency cutoff for common tokens |
| `--batch-size` | `500` | S1 queries per sparse matrix multiplication |
| `--chunk-size` | `0` | If > 0, process S1 in chunks of this size and stream results to disk (per-entity results are identical to the single-call mode) |
| `--workers` | `1` | Forked worker processes for chunked mode; a value > 1 enables chunked mode (default chunk size 10,000) |

Notes:

- The detailed parquet is written incrementally and is **only readable once the run has finished** (the file footer is written on close). `candidate_pairs.tsv` is valid at any point but only covers the S1 entities processed so far, so an interrupted run does **not** produce a submittable file.
- Runs are not resumable; an interrupted run must be restarted.
- `output/` is **not** listed in `.gitignore` (only `outputs/` is) and a full run produces a multi-GB parquet, so do not commit it.

**Quick smoke test** on a small slice (e.g. a few thousand S1 rows plus a matching S2/S3 subset) before any long run: check that the output has exactly the columns listed in section 1.

---

## 5. Submission Validation

Verify generated files against the official challenge validator:

```bash
python student_resource/utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir student_resource/dataset/test
```

(`matching_results.tsv` comes from the later matching stage; this module only produces `candidate_pairs.tsv`.)

---

## 6. Scale, runtime and memory

Measured on the full test set (1,732,544 S1 entities; 9,969,589 S2 + S3 records; `--top-k 50`) on a 16 GB Apple-silicon Mac with 4 performance + 6 efficiency cores:

| Item | Measured |
| :--- | :--- |
| Index build (`fit`) | ≈ 5–6 min, ≈ 6 GB RAM |
| Generation, single process | ≈ 52 ms per S1 entity → ≈ 25 h for the full test set |
| Generation, `--workers 6` | ≈ 130 ms per S1 entity per worker (≈ 46 entities/s overall); on average ≈ 3.7 min per completed 10,000-entity chunk (6 in flight; measured over the first 18 of 174 chunks) → **≈ 10–11 h** for the full test set |
| Expected output size | ≈ 86.6 million candidate pairs (≈ 50 per S1 entity) |
| In-memory mode (no chunking) | ≈ 0.4 GB per 200k candidate rows, i.e. tens of GB at full scale — use `--chunk-size` / `--workers` |

- Retrieval is memory-bandwidth-bound sparse matrix multiplication, so parallel speed-up is well below linear (≈ 2.4× with 6 workers here).
- Keep the machine awake for the whole run (e.g. `caffeinate -i -w <PID>` on macOS).
- **Recall:** on a small smoke test (2,000 train S1 entities against a 107k-record pool) recall of true matches was 98.9%, which is optimistic because the pool is tiny. Full-pool recall of this implementation has not been measured; an earlier prototype of the same retrieval idea measured ≈ 93.5% (US) / 87.0% (India) at K = 20 and ≈ 95.7% / 90.7% at K = 100 on training data.
