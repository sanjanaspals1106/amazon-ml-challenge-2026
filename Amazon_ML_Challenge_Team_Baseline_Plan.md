# Amazon ML Challenge 2026 — Team Execution Plan
## Baseline Phase: Team Split, Timeline, Git Workflow, Model Specifications & Integration

**Project repository:** `amazon-ml-challenge-2026`  
**Local project:** `MLChallenge/`  
**Competition window:** 25 Sep 2026 – 27 Sep 2026  
**Team size:** 3  
**Current repository state:** Initial reconnaissance + project structure committed to `main`

---

# 1. Goal of This Document

This document defines the team's execution plan **up to the first strong baseline model**.

The purpose is to avoid three people independently building overlapping solutions. We will first build one complete, reproducible pipeline:

```text
Raw S1 / S2 / S3
      ↓
Normalization
      ↓
Candidate Generation / Blocking
      ↓
Pairwise Features
      ↓
Baseline Matcher
      ↓
One-owner / threshold post-processing
      ↓
matching_results.tsv
candidate_pairs.tsv
      ↓
Local validation
      ↓
Submission
```

Only after this baseline is working and evaluated will we decide whether additional models, embeddings, hard-negative mining, or more advanced approaches are worth the remaining submission/engineering time.

---

# 2. Important Findings Driving the Baseline

The dataset reconnaissance already established several important facts:

- Exact normalized name/address matching is insufficient. A large fraction of true pairs do not share an exact normalized name or address.
- Name and address are complementary signals.
- Address is missing for a minority of positives, so the model must handle missing address rather than assuming address always exists.
- Simple blocking keys such as first-token/name-prefix rules can generate very large candidate sets or lose many true matches.
- IDF-weighted token retrieval gives substantially better candidate recall than naive blocking.
- At K=100, diagnostic candidate recall was approximately 95.7% for US and 90.6–90.7% for India; K=50 was approximately 95.2% US and 89.3% India. These are retrieval diagnostics, not final model scores.
- A diagnostic gradient-boosting model reached approximately 0.89 macro F0.5 on realistic candidate lists, with stronger US performance than India. This is a diagnostic baseline, not a final competition model.
- Hard negatives include sibling/look-alike entities sharing buildings or addresses but differing in unit/plot/suffix information.
- Each S2/S3 record belongs to at most one S1 entity in the ground truth. This supports a one-owner post-processing step.
- France is an unseen test country and therefore must not be treated as merely another US/India case. France needs an open-set-aware normalization/retrieval strategy, but it is NOT the first implementation priority.
- Validation must be grouped by S1 entity rather than randomly splitting candidate-pair rows.

Source: `dataset_analysis_report.md`, especially the sections covering candidate generation, pair features, model diagnostics, validation, and France/test shift.

---

# 3. Team Responsibilities

## Person 1 — You: Architecture + Integration + Final Pipeline

### Primary ownership

You own the **overall pipeline and integration**.

Your areas:

```text
src/pipeline.py
src/postprocessing/
configs/
experiments/
submission generation
integration testing
```

You also own:

- final threshold selection
- experiment log
- local validation orchestration
- France/open-set strategy after baseline
- one-owner constraint
- final submission files
- GitHub `main`
- merging teammates' branches

### You should NOT duplicate

Do not independently rebuild the blocking algorithm or the classifier while teammates are implementing them.

Your job is to define the interfaces and make their components plug together.

---

## Person 2 — Teammate A: Preprocessing + Candidate Generation / Blocking

### Primary ownership

```text
src/preprocessing/
src/blocking/
```

### Goal

Build a candidate generator that dramatically reduces the S1 × S2/S3 search space while retaining as many true matches as possible.

### Required baseline approach

Implement a country-aware candidate generator using a UNION of:

1. Exact normalized name
2. Exact normalized address
3. IDF-weighted address-token retrieval
4. IDF-weighted name-token retrieval
5. Name 4-character prefix retrieval
6. Native-script/address-oriented fallback where needed

The first benchmark should test:

```text
K = 20
K = 50
K = 100
```

### Required outputs

The module should produce candidate pairs with at least:

```text
s1_id
source
s2_or_s3_id
country
retrieval_rank
retrieval_score
retrieval_method
```

The output must be consumable by Person 3.

### Required measurements

For each K:

- candidate count
- average candidates per S1
- true-match recall
- entity/all-matches recall where measurable
- US recall
- India recall
- runtime
- memory usage

### Platform

**Primary platform: LAPTOP**

Why:

- The raw dataset is already local.
- Candidate generation is CPU/data-processing heavy.
- Moving large candidate sets repeatedly to Colab is unnecessary.
- We need reproducible local integration.

Do NOT use Colab for the first blocking implementation.

### Deliverable

```text
feature/blocking
    ↓
blocking module
    ↓
benchmark report
    ↓
candidate output schema
    ↓
commit + push
```

---

# 4. Person 3 — Teammate B: Pair Features + Baseline Matcher

### Primary ownership

```text
src/features/
src/model/
```

### Goal

Build the first supervised pair classifier using **realistic candidate pairs generated by Person 2**.

Do NOT train using only random negatives. Random negatives are too easy and do not represent the sibling/look-alike false positives found in the data.

---

# 5. Baseline Feature Specification

The baseline should use approximately 30–40 features.

## Name features

Include:

- normalized name token-sort similarity
- normalized name token-set similarity
- partial similarity
- Jaro-Winkler similarity
- normalized Levenshtein similarity
- token Jaccard
- token overlap
- character 3-gram TF-IDF cosine similarity
- first-token equality
- token-count difference
- character-length difference
- core-name similarity
- legal-form agreement
- legal-form conflict
- initials-only flag
- alias/domain-name flag

## Address features

Include:

- address token-set similarity
- address token Jaccard
- IDF-weighted address overlap
- character 3-gram TF-IDF cosine
- house-number equality
- house-number equality after leading-zero normalization
- house-number difference/off-by-one signal
- city agreement
- state/region agreement
- number of missing/extra address components
- empty-address indicators

## Context features

Include:

- country
- source: S2 or S3
- retrieval rank
- retrieval score
- retrieval method
- S1 name multiplicity
- candidate competition / number of S1 entities competing for the same source record
- whether this S1 is the candidate's best-scoring owner

---

# 6. Baseline Model Specification

## Model

Start with a **gradient-boosted tree classifier**.

Preferred order:

### Option A — XGBoost

```text
XGBClassifier
```

Use this if the environment is already working reliably.

### Option B — sklearn HistGradientBoostingClassifier

Use this as the fallback because it avoids adding unnecessary dependency/debugging complexity.

We do NOT need a neural network for the baseline.

We do NOT train a Transformer.

We do NOT start with sentence-transformer embeddings.

Embeddings are a later experiment only if the baseline shows a specific weakness that they could address.

---

# 7. Training Data Construction

The classifier must train on:

```text
S1
 ↓
retriever
 ↓
candidate list
 ↓
candidate pair
 ↓
label = 1/0
 ↓
feature vector
```

Positive pairs:

```text
candidate pair corresponds to the known ground-truth S1 owner
```

Negative pairs:

```text
retrieved candidates that are not the true owner
```

This naturally creates realistic hard negatives.

Include sibling/look-alike candidates where available.

---

# 8. Validation Split

Do NOT randomly split candidate rows.

Split by **S1 entity**.

Example:

```text
Train S1 entities
Validation S1 entities
```

This prevents candidate pairs belonging to the same S1 entity from leaking between train and validation.

Report:

- macro F0.5
- precision
- recall
- US F0.5
- India F0.5
- S2 F0.5
- S3 F0.5
- singleton S1 performance
- non-singleton performance
- Latin/native-script breakdown where meaningful

---

# 9. Where the Model Should Be Trained

## Baseline recommendation

### Candidate generation

**LAPTOP**

```text
Raw dataset
   ↓
normalization
   ↓
blocking
   ↓
candidate pairs
```

### Feature extraction

**LAPTOP initially**

Generate the feature table locally from the candidate pairs.

### Model training

**COLAB is OPTIONAL for the first model, not mandatory.**

Recommended workflow:

```text
LAPTOP
    ↓
generate candidate pairs
    ↓
generate feature dataset
    ↓
save compact training/validation feature files
    ↓
COLAB
    ↓
train/test baseline model
    ↓
save model + metrics
    ↓
LAPTOP
    ↓
integration
```

If the feature dataset is small enough and local training is fast, train locally instead. There is no requirement to use Colab just because it is available.

### AWS

**NOT required for the baseline.**

Do not spend the first phase setting up AWS.

AWS becomes an option only if:

- full-scale retrieval is too slow locally,
- model training becomes computationally expensive,
- the challenge explicitly requires AWS infrastructure,
- or challenge-provided compute credits give a clear advantage.

---

# 10. Baseline Integration Contract

Person 2 must expose something conceptually like:

```python
generate_candidates(
    s1_df,
    source_df,
    config
) -> candidate_df
```

Output:

```text
s1_id
source_record_id
source
country
retrieval_rank
retrieval_score
retrieval_method
```

Person 3 must expose:

```python
build_features(
    candidate_df,
    s1_df,
    source_df,
    config
) -> feature_df
```

And:

```python
train_matcher(
    X_train,
    y_train,
    config
) -> model
```

Prediction:

```python
predict_match_scores(
    model,
    feature_df
) -> scores
```

You then integrate them in:

```text
src/pipeline.py
```

---

# 11. Baseline Post-processing

After the classifier produces probabilities/scores:

```text
S1 A → S2-123 = 0.91
S1 B → S2-123 = 0.87
```

Use the dataset property that each S2/S3 record has at most one S1 owner.

For the baseline:

1. Score all candidates.
2. Apply a threshold.
3. For competing S1 assignments to the same S2/S3 record, prefer the strongest assignment.
4. Drop near-ties when confidence is insufficient.
5. Allow zero matches.
6. Allow multiple matches for an S1.

Do not force every S1 to have a match.

Singleton S1 entities exist, and the analysis found that singleton status is not reliably predictable from S1-side attributes. Therefore "no match" must be based on evidence rather than an S1-side rule.

---

# 12. Threshold Selection

Do not choose the threshold arbitrarily.

Evaluate a grid such as:

```text
0.30
0.40
0.50
0.55
0.60
0.65
0.70
0.75
0.80
0.85
0.90
```

For each threshold calculate:

```text
macro F0.5
precision
recall
false merges
```

Because the competition metric is precision-weighted, false merges need particular attention.

The diagnostic model suggested a fairly broad useful region around approximately 0.6–0.7, but the actual final threshold must be selected from our grouped validation results.

---

# 13. Timeline

## DAY 1 — 25 September

### Phase 1 — Repository + interfaces

**You**

- finalize repo structure
- create configs
- define candidate schema
- define feature schema
- create experiment log
- create pipeline skeleton

**Person 2**

- implement normalization
- implement candidate generation
- benchmark K=20/50/100

**Person 3**

- implement feature extraction
- prepare classifier interface
- build training dataset from Person 2's candidate output
- train first gradient-boosting diagnostic

### Day 1 checkpoint

By end of Day 1 we want:

```text
Raw data
  ↓
Normalization
  ↓
Candidate generation
  ↓
Features
  ↓
Baseline classifier
```

with at least one complete local validation run.

---

# 14. Git Merge Plan

## Main branch rule

`main` is always the integrated/stable branch.

Nobody develops directly on `main`.

Branches:

```text
main

feature/integration
feature/blocking
feature/matching-model
```

---

## Person 2 workflow

```bash
git checkout feature/blocking
git pull origin main

# work

git add .
git commit -m "Implement baseline candidate generation"
git push origin feature/blocking
```

Then open a Pull Request:

```text
feature/blocking → main
```

Do NOT merge until the candidate schema and benchmark results are documented.

---

## Person 3 workflow

```bash
git checkout feature/matching-model
git pull origin main

# work

git add .
git commit -m "Implement baseline matching features and model"
git push origin feature/matching-model
```

Open:

```text
feature/matching-model → main
```

This should be merged after the candidate interface is stable.

---

## Your workflow

You maintain:

```text
feature/integration
```

You can build the pipeline around the agreed interfaces without waiting for every internal implementation.

Before merging:

```bash
git checkout main
git pull origin main

git merge feature/blocking
git merge feature/matching-model
```

Then run the full baseline validation.

Only after it works:

```bash
git push origin main
```

---

# 15. Merge Order

Use this order:

```text
1. Repository structure
       ↓
2. Blocking interface
       ↓
3. Baseline candidate generator
       ↓
4. Candidate benchmark
       ↓
5. Feature extractor
       ↓
6. Baseline classifier
       ↓
7. Integration
       ↓
8. Threshold + post-processing
       ↓
9. Local validation
       ↓
10. First baseline submission
```

Do not merge large unrelated changes together.

---

# 16. Experiment Tracking

Create:

```text
experiments/experiment_log.csv
```

Columns:

```text
experiment_id
date
owner
blocking
K
normalization
features
model
threshold
validation_f05
precision
recall
US_f05
India_f05
S2_f05
S3_f05
notes
```

Every meaningful experiment gets one row.

This is especially important because the challenge has limited submission opportunities. We should not use submissions as random experiments.

---

# 17. Baseline Definition

Our first baseline is considered COMPLETE only when all of the following exist:

### Data

- [ ] S1/S2/S3 loading works
- [ ] normalization works
- [ ] country handling works

### Candidate generation

- [ ] candidate generator works
- [ ] K=20/50/100 benchmark exists
- [ ] candidate recall measured
- [ ] runtime measured

### Features

- [ ] name features
- [ ] address features
- [ ] legal-form features
- [ ] house-number features
- [ ] retrieval/context features
- [ ] missing-value handling

### Model

- [ ] gradient boosting classifier
- [ ] realistic negatives
- [ ] grouped S1 validation
- [ ] probability scores

### Post-processing

- [ ] threshold
- [ ] one-owner rule
- [ ] zero-match handling
- [ ] multiple-match handling

### Outputs

- [ ] `matching_results.tsv`
- [ ] `candidate_pairs.tsv`
- [ ] validation metrics
- [ ] experiment record
- [ ] reproducible command

---

# 18. What Happens After the Baseline?

DO NOT implement these before the baseline is measured:

### Possible Phase 2 experiments

1. Hard-negative mining
2. Better sibling/look-alike discrimination
3. France-specific normalization
4. Source-specific thresholds/features
5. Character n-gram retrieval
6. Sentence-transformer embeddings
7. FAISS vector retrieval
8. Ensemble of lexical + embedding scores
9. Better global one-owner optimization
10. Additional normalization discovered from validation errors

The decision should be based on **where the baseline fails**, not on adding complexity for its own sake.

---

# 19. France Strategy — Phase 2

France is an unseen test country, so we should not overfit the first baseline to US/India assumptions.

For the first baseline:

- country remains an explicit feature
- do not use ZIP/PIN as a required feature
- normalization should be country-aware
- do not hard-code US/India as the only supported countries

After baseline:

### France-specific improvements

- accent folding
- French street abbreviation normalization
- French legal-form extraction
- handling of `R.`, `Rue`, `Av`, `Avenue`, `Bd`, `Bvd`, `Pl.`
- handling of `SARL`, `SAS`, `EURL`, `SASU`, `SCI`, `SA`
- French stop-token treatment such as `de`, `du`, `des`, `la`, `le`
- evaluation using US→India / India→US leave-one-country-out experiments as a proxy for unseen-country behavior

Do not assume these changes improve performance until validated.

---

# 20. Platform Decision Summary

| Work | Platform | Reason |
|---|---|---|
| Git/GitHub | GitHub | Single source of truth |
| Dataset storage | Laptop | Dataset is large; do not commit it |
| Reconnaissance | Laptop | Already completed |
| Normalization | Laptop | CPU/data processing |
| Candidate generation | **Laptop** | Large-scale retrieval; avoid unnecessary data transfer |
| Candidate benchmarking | **Laptop** | Needs access to full local dataset |
| Feature generation | **Laptop** | Same data locality |
| Baseline model training | **Laptop first; Colab optional** | Start simple; use Colab only if training benefits |
| Model experiments | Colab optional | Useful for isolated ML experiments |
| Pipeline integration | **Laptop** | Final reproducible pipeline |
| Final output generation | **Laptop** | Same environment as pipeline |
| AWS | **Not required for baseline** | Consider only if scale/runtime requires it |
| Submission | Challenge portal | Final TSV files |

---

# 21. Critical Rules

### Rule 1 — One repository

No separate codebases.

### Rule 2 — Dataset is never committed

Never run:

```bash
git add dataset.zip
```

### Rule 3 — No direct development on main

Use feature branches.

### Rule 4 — Interfaces before implementations

Agree on candidate and feature schemas before teammates connect their code.

### Rule 5 — No random-negative-only model

Use retrieved candidates.

### Rule 6 — Validate by S1 entity

Do not randomly split candidate rows.

### Rule 7 — Baseline before advanced models

Do not jump to embeddings/deep learning before measuring the first complete pipeline.

### Rule 8 — Every experiment is recorded

No "I changed three things and the score went up" experiments.

### Rule 9 — Local validation before submission

A submission should be the result of a deliberate experiment.

### Rule 10 — Optimize precision carefully

False merges are particularly damaging for this task.

---

# 22. First Commands for Each Person

## You

```bash
git checkout feature/integration
git pull origin main
```

Start with:

```text
src/pipeline.py
src/postprocessing/
configs/
experiments/experiment_log.csv
```

## Person 2

```bash
git checkout -b feature/blocking
git push -u origin feature/blocking
```

Start with:

```text
src/preprocessing/
src/blocking/
```

## Person 3

```bash
git checkout -b feature/matching-model
git push -u origin feature/matching-model
```

Start with:

```text
src/features/
src/model/
```

---

# 23. Definition of Done for the First Baseline

The team stops Phase 1 when we can execute something conceptually like:

```bash
python -m src.pipeline \
    --config configs/baseline.yaml
```

and obtain:

```text
outputs/
├── matching_results.tsv
├── candidate_pairs.tsv
└── baseline_metrics.json
```

with a reproducible validation score.

At that point, **we stop and inspect errors before choosing the next model**.

