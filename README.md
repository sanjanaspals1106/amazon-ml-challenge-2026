# Amazon ML Challenge 2026

Team solution for the Amazon ML Challenge 2026.

## Problem

Entity resolution / record linkage across three sources.

For every entity in Source 1, identify matching entities from Source 2
and Source 3.

## Approach

The planned pipeline consists of:

1. Data preprocessing and normalization
2. Candidate generation / blocking
3. Pairwise feature extraction
4. Matching model
5. Post-processing and one-owner constraints
6. Submission generation

## Repository Structure

```text
src/
├── preprocessing/
├── blocking/
├── features/
├── model/
└── postprocessing/

analysis/
experiments/
configs/
notebooks/
