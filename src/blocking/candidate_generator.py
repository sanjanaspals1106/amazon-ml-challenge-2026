"""
Candidate Generator (Blocking Pipeline) for Amazon ML Challenge 2026.
Role: Person 2 (Teammate A).

Produces candidates for each Source 1 record from the combined Source 2 and Source 3 corpus.
Output DataFrame schema:
- s1_id: Source 1 entity ID
- source_record_id: Candidate entity ID from Source 2 or Source 3
- source: 'S2' or 'S3'
- country: Country of the record
- retrieval_rank: 1-indexed rank of candidate for this S1 entity (1, 2, ..., K)
- retrieval_score: Similarity / overlap score
- retrieval_method: Retrieval channel ('exact_name', 'exact_addr', 'tfidf_name', 'tfidf_addr', 'union')
"""

import os
import re
import time
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer

from .text_processor import (
    extract_prefixes_4,
    normalize_address,
    normalize_name,
    sort_tokens,
)


class CandidateGenerator:
    """
    Multi-channel candidate generator and blocking engine.
    Partitions the search space by country and combines:
    1. Exact Key Channel: Token-sorted name and address exact match
    2. Name TF-IDF Channel: Token + 4-char prefix IDF-weighted retrieval
    3. Address TF-IDF Channel: Address token IDF-weighted retrieval
    """

    def __init__(
        self,
        top_k: int = 50,
        df_max: int = 20000,
        batch_size: int = 500,
        prefix_weight: float = 0.5,
        verbose: bool = True,
        exact_name_cap: int = 8,
        exact_name_boost: float = 50.0,
    ):
        """
        Args:
            top_k: Maximum candidates to retain per S1 entity.
            df_max: Document frequency threshold above which tokens are suppressed (stops common noise words).
            batch_size: Number of S1 queries processed in a single sparse matrix multiplication batch.
            prefix_weight: Weight discount for 4-character name prefixes vs full tokens.
            verbose: If True, prints progress logs.
            exact_name_cap: Maximum number of exact_name-tier candidates kept per S1 entity.
                Exact-name matches are common (many businesses share a normalized name) but only
                weakly predictive on their own, so they are ranked by their own sparse
                name/address overlap score instead of a flat boost, and only the top
                `exact_name_cap` are kept -- this frees up top-K slots for exact_addr /
                sparse_tfidf candidates that were previously being crowded out.
            exact_name_boost: Small additive boost applied on top of an exact_name candidate's
                sparse score, so exact-name ties break above pure sparse_tfidf candidates with a
                similar score, without letting them dominate the full 100.0 flat score used before.
        """
        self.top_k = top_k
        self.df_max = df_max
        self.batch_size = batch_size
        self.prefix_weight = prefix_weight
        self.verbose = verbose
        self.exact_name_cap = exact_name_cap
        self.exact_name_boost = exact_name_boost

        # Country-indexed models & corpora
        self.country_models: Dict[str, Dict] = {}
        self.is_fitted: bool = False

    def _log(self, msg: str):
        if self.verbose:
            print(f"[CandidateGenerator] {msg}", flush=True)

    def fit(
        self,
        source2_df: pd.DataFrame,
        source3_df: pd.DataFrame,
    ) -> "CandidateGenerator":
        """
        Fit vocabulary, IDF weights, and exact lookup indices on the candidate corpus (Source 2 + Source 3).
        Expects columns: ['entity_id', 'business_name', 'business_address', 'country']
        """
        t0 = time.time()
        self._log("Preprocessing and indexing Source 2 and Source 3 corpora...")

        # Add source identifier
        s2 = source2_df[["entity_id", "business_name", "business_address", "country"]].copy()
        s2["source"] = "S2"
        s3 = source3_df[["entity_id", "business_name", "business_address", "country"]].copy()
        s3["source"] = "S3"

        corpus = pd.concat([s2, s3], ignore_index=True)
        self._log(f"Total candidate pool: {len(corpus):,} records ({len(s2):,} S2, {len(s3):,} S3)")

        # Normalize text fields
        self._log("Normalizing corpus names and addresses...")
        corpus["norm_name"] = corpus["business_name"].apply(normalize_name)
        corpus["norm_addr"] = corpus["business_address"].apply(normalize_address)
        corpus["sorted_name"] = corpus["norm_name"].apply(sort_tokens)
        corpus["sorted_addr"] = corpus["norm_addr"].apply(sort_tokens)
        corpus["name_p4"] = corpus["norm_name"].apply(extract_prefixes_4)

        # Build country-specific indices
        countries = corpus["country"].unique()
        self._log(f"Building per-country indices for {len(countries)} countries: {list(countries)}")

        for country in countries:
            c_mask = corpus["country"] == country
            c_df = corpus[c_mask].reset_index(drop=True)
            n_docs = len(c_df)
            self._log(f"Indexing country '{country}' with {n_docs:,} records...")

            # 1. Exact lookup hash tables (sorted name -> [corpus_indices], sorted addr -> [corpus_indices])
            name_exact_map: Dict[str, List[int]] = {}
            for idx, sn in enumerate(c_df["sorted_name"]):
                if sn:
                    name_exact_map.setdefault(sn, []).append(idx)

            addr_exact_map: Dict[str, List[int]] = {}
            for idx, sa in enumerate(c_df["sorted_addr"]):
                if sa:
                    addr_exact_map.setdefault(sa, []).append(idx)

            # 2. Sparse Vectorizers & Inverted Indices
            # Name channel
            vec_name = CountVectorizer(token_pattern=r"\S+", binary=True, dtype=np.float32, lowercase=False)
            X_name = vec_name.fit_transform(c_df["norm_name"]).tocsr()
            df_name = np.asarray(X_name.sum(0)).ravel()
            with np.errstate(divide="ignore", invalid="ignore"):
                idf_name = np.log(n_docs / np.maximum(df_name, 1.0)).astype(np.float32)
            idf_name[df_name > self.df_max] = 0.0

            # Prefix channel
            vec_p4 = CountVectorizer(token_pattern=r"\S+", binary=True, dtype=np.float32, lowercase=False)
            X_p4 = vec_p4.fit_transform(c_df["name_p4"]).tocsr()
            df_p4 = np.asarray(X_p4.sum(0)).ravel()
            with np.errstate(divide="ignore", invalid="ignore"):
                idf_p4 = np.log(n_docs / np.maximum(df_p4, 1.0)).astype(np.float32)
            idf_p4[df_p4 > self.df_max] = 0.0

            # Address channel
            vec_addr = CountVectorizer(token_pattern=r"\S+", binary=True, dtype=np.float32, lowercase=False)
            X_addr = vec_addr.fit_transform(c_df["norm_addr"]).tocsr()
            df_addr = np.asarray(X_addr.sum(0)).ravel()
            with np.errstate(divide="ignore", invalid="ignore"):
                idf_addr = np.log(n_docs / np.maximum(df_addr, 1.0)).astype(np.float32)
            idf_addr[df_addr > self.df_max] = 0.0

            self.country_models[country] = {
                "corpus_df": c_df[["entity_id", "source", "country"]],
                "name_exact_map": name_exact_map,
                "addr_exact_map": addr_exact_map,
                "vec_name": vec_name,
                "XT_name": X_name.T.tocsr(),
                "idf_name": idf_name,
                "vec_p4": vec_p4,
                "XT_p4": X_p4.T.tocsr(),
                "idf_p4": idf_p4 * self.prefix_weight,
                "vec_addr": vec_addr,
                "XT_addr": X_addr.T.tocsr(),
                "idf_addr": idf_addr,
            }

        self.is_fitted = True
        self._log(f"Candidate index successfully built in {time.time() - t0:.1f}s.")
        return self

    def generate_candidates(
        self,
        source1_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Generate candidate pairs for all entities in Source 1.

        Args:
            source1_df: DataFrame with ['entity_id', 'business_name', 'business_address', 'country']

        Returns:
            pd.DataFrame with columns:
            ['s1_id', 'source_record_id', 'source', 'country', 'retrieval_rank', 'retrieval_score', 'retrieval_method']
        """
        if not self.is_fitted:
            raise RuntimeError("CandidateGenerator must be fitted with fit() before calling generate_candidates().")

        t0 = time.time()
        self._log(f"Generating candidates for {len(source1_df):,} Source 1 records...")

        # Normalize S1 records
        s1 = source1_df[["entity_id", "business_name", "business_address", "country"]].copy()
        s1["norm_name"] = s1["business_name"].apply(normalize_name)
        s1["norm_addr"] = s1["business_address"].apply(normalize_address)
        s1["sorted_name"] = s1["norm_name"].apply(sort_tokens)
        s1["sorted_addr"] = s1["norm_addr"].apply(sort_tokens)
        s1["name_p4"] = s1["norm_name"].apply(extract_prefixes_4)

        results: List[Dict] = []
        countries = s1["country"].unique()

        for country in countries:
            c_s1 = s1[s1["country"] == country].reset_index(drop=True)
            n_queries = len(c_s1)
            self._log(f"Processing {n_queries:,} Source 1 records for country '{country}'...")

            if country not in self.country_models:
                self._log(f"Warning: Country '{country}' has no matching candidate corpus in Source 2/3.")
                continue

            model = self.country_models[country]
            c_corpus = model["corpus_df"]
            c_entity_ids = c_corpus["entity_id"].values
            c_sources = c_corpus["source"].values

            name_exact_map = model["name_exact_map"]
            addr_exact_map = model["addr_exact_map"]

            vec_name, XT_name, idf_name = model["vec_name"], model["XT_name"], model["idf_name"]
            vec_p4, XT_p4, idf_p4 = model["vec_p4"], model["XT_p4"], model["idf_p4"]
            vec_addr, XT_addr, idf_addr = model["vec_addr"], model["XT_addr"], model["idf_addr"]

            # Process in batches
            for st in range(0, n_queries, self.batch_size):
                batch = c_s1.iloc[st : st + self.batch_size]
                b_size = len(batch)
                b_ids = batch["entity_id"].values
                b_sn = batch["sorted_name"].values
                b_sa = batch["sorted_addr"].values

                # Sparse scoring
                # NOTE: .multiply(idf) preserves the sparsity pattern of the input, so tokens
                # with df > df_max (idf forced to 0.0, e.g. "private", "llc") are kept as
                # *explicit* stored zeros. A CSR @ CSR matmul still walks the full posting list
                # for every stored entry regardless of its value, so those zero-weight columns
                # were silently costing the majority of the retrieval time. eliminate_zeros()
                # drops those explicit zeros so the matmul skips them entirely. This is a pure
                # sparsity-pruning step: any entry with weight 0 contributes 0 to the dot product
                # either way, so scores and rankings are unchanged.
                # 1. Name query
                A_name = vec_name.transform(batch["norm_name"]).tocsr()
                A_name = A_name.multiply(idf_name).tocsr()
                A_name.eliminate_zeros()
                P_name = A_name @ XT_name

                # 2. Prefix query
                A_p4 = vec_p4.transform(batch["name_p4"]).tocsr()
                A_p4 = A_p4.multiply(idf_p4).tocsr()
                A_p4.eliminate_zeros()
                P_p4 = A_p4 @ XT_p4

                # 3. Address query
                A_addr = vec_addr.transform(batch["norm_addr"]).tocsr()
                A_addr = A_addr.multiply(idf_addr).tocsr()
                A_addr.eliminate_zeros()
                P_addr = A_addr @ XT_addr

                # Summed sparse scores
                P_total = (P_name + P_p4 + P_addr).tocsr()

                # Process per row in batch
                for row_idx in range(b_size):
                    s1_id = b_ids[row_idx]
                    sn = b_sn[row_idx]
                    sa = b_sa[row_idx]

                    candidate_dict: Dict[int, Tuple[float, str]] = {}

                    # 2. Sparse TF-IDF scores (computed before the exact tiers so exact_name
                    #    candidates can be ranked/capped using their own overlap score instead
                    #    of a flat boost).
                    p_start = P_total.indptr[row_idx]
                    p_end = P_total.indptr[row_idx + 1]
                    indices = P_total.indices[p_start:p_end]
                    scores = P_total.data[p_start:p_end]
                    sparse_score_map: Dict[int, float] = (
                        dict(zip(indices.tolist(), scores.tolist())) if len(indices) > 0 else {}
                    )

                    # 1a. Exact Matches Tier -- exact_addr / exact_both.
                    # These are highly precise (~77-100% true-match rate) and are NOT capped:
                    # they keep the original flat boost and always win their top-K slot.
                    if sa and sa in addr_exact_map:
                        for c_idx in addr_exact_map[sa]:
                            if sn and sn in name_exact_map and c_idx in name_exact_map.get(sn, ()):
                                candidate_dict[c_idx] = (110.0, "exact_both")
                            else:
                                candidate_dict[c_idx] = (95.0, "exact_addr")

                    # 1b. Exact Matches Tier -- exact_name only.
                    # Exact-name matches alone are common but weakly predictive (~10.7% true
                    # match rate measured on dev), so a flat 100.0 boost let them flood the
                    # top-K and crowd out exact_addr / sparse_tfidf candidates. Instead: rank
                    # exact-name matches by their own sparse retrieval score and keep only the
                    # top `exact_name_cap` of them, each getting a small additive boost so ties
                    # break in their favor without dominating the list.
                    if sn and sn in name_exact_map:
                        en_candidates = [
                            (c_idx, sparse_score_map.get(c_idx, 0.0))
                            for c_idx in name_exact_map[sn]
                            if c_idx not in candidate_dict  # skip ones already exact_addr/both
                        ]
                        en_candidates.sort(key=lambda item: item[1], reverse=True)
                        for c_idx, base_score in en_candidates[: self.exact_name_cap]:
                            candidate_dict[c_idx] = (base_score + self.exact_name_boost, "exact_name")

                    if len(indices) > 0:
                        # Top candidates selection
                        if len(indices) > self.top_k:
                            top_partition = np.argpartition(-scores, self.top_k)[: self.top_k]
                            indices = indices[top_partition]
                            scores = scores[top_partition]

                        for c_idx, score in zip(indices, scores):
                            if c_idx not in candidate_dict or candidate_dict[c_idx][0] < score:
                                candidate_dict[c_idx] = (float(score), "sparse_tfidf")

                    # Sort all candidates for this S1 by score descending
                    if candidate_dict:
                        sorted_cands = sorted(
                            candidate_dict.items(), key=lambda item: item[1][0], reverse=True
                        )[: self.top_k]

                        for rank, (c_idx, (score, method)) in enumerate(sorted_cands, start=1):
                            results.append(
                                {
                                    "s1_id": s1_id,
                                    "source_record_id": c_entity_ids[c_idx],
                                    "source": c_sources[c_idx],
                                    "country": country,
                                    "retrieval_rank": rank,
                                    "retrieval_score": round(float(score), 4),
                                    "retrieval_method": method,
                                }
                            )

        output_df = pd.DataFrame(
            results,
            columns=[
                "s1_id",
                "source_record_id",
                "source",
                "country",
                "retrieval_rank",
                "retrieval_score",
                "retrieval_method",
            ],
        )
        self._log(
            f"Generated {len(output_df):,} total candidate pairs for {len(source1_df):,} S1 entities in {time.time() - t0:.1f}s."
        )
        return output_df

    @staticmethod
    def export_candidate_pairs_tsv(
        candidates_df: pd.DataFrame,
        all_s1_ids: Union[List[str], pd.Series],
        output_path: str,
    ) -> str:
        """
        Export candidates into official submission format:
        `output/candidate_pairs.tsv` with columns:
        `source1_entity_id\\tcandidate_entity_ids` (comma-separated).

        Ensures:
        - Every S1 entity in all_s1_ids appears exactly once.
        - S1 with no candidates have empty candidate_entity_ids.
        - No duplicate candidate IDs per list.
        - Tab-separated TSV with UTF-8 encoding.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        # Group candidates by S1 ID
        cand_map: Dict[str, List[str]] = {}
        if not candidates_df.empty:
            for s1_id, group in candidates_df.groupby("s1_id", sort=False):
                # Preserve rank order and remove any duplicates
                unique_cands = list(dict.fromkeys(group["source_record_id"].tolist()))
                cand_map[s1_id] = unique_cands

        # Build rows for all requested S1 IDs
        rows = []
        for s1_id in all_s1_ids:
            cands = cand_map.get(s1_id, [])
            cands_str = ",".join(cands)
            rows.append(f"{s1_id}\t{cands_str}\n")

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("source1_entity_id\tcandidate_entity_ids\n")
            f.writelines(rows)

        return output_path
