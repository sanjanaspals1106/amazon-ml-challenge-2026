"""Tests for P1 modules: one-owner rule, entity-level metrics, threshold search, grouped split, writers, logging."""

import numpy as np
import pandas as pd
import pytest

from src.experiment_log import COLUMNS, append_experiment
from src.model.training import compute_entity_f05  # P3's per-entity reference implementation
from src.postprocessing import (
    evaluate_entities, evaluate_with_segments, grouped_split, make_threshold_grid, resolve_one_owner,
    search_threshold, truth_pairs, verify_one_owner,
)
from src.postprocessing.output import MATCHING_HEADER, verify_output_file, write_matching_results


def _scored(rows):
    return pd.DataFrame(rows, columns=["s1_id", "source_record_id", "score"])


# ---------------------------------------------------------------- one-owner
def test_one_owner_strongest_wins_and_near_tie_dropped():
    df = _scored([("A", "S2-1", 0.91), ("B", "S2-1", 0.87), ("A", "S2-2", 0.80), ("C", "S3-1", 0.95), ("D", "S3-9", 0.20)])
    kept, info = resolve_one_owner(df, threshold=0.5, near_tie_margin=0.01)
    assert set(map(tuple, kept[["s1_id", "source_record_id"]].values)) == {("A", "S2-1"), ("A", "S2-2"), ("C", "S3-1")}
    assert info["n_contested_records"] == 1 and info["n_near_tie_dropped_records"] == 0
    # gap 0.04 <= margin 0.05 -> S2-1 dropped for both claimants; A keeps S2-2 (one S1 may own many records)
    kept2, info2 = resolve_one_owner(df, threshold=0.5, near_tie_margin=0.05)
    assert set(kept2["source_record_id"]) == {"S2-2", "S3-1"} and info2["n_near_tie_dropped_records"] == 1
    verify_one_owner(kept2)


def test_one_owner_margin_none_keeps_best_and_breaks_exact_tie_deterministically():
    df = _scored([("B", "S2-1", 0.8), ("A", "S2-1", 0.8), ("C", "S2-2", 0.9), ("D", "S2-2", 0.6)])
    kept, _ = resolve_one_owner(df, 0.5, near_tie_margin=None)
    assert dict(zip(kept["source_record_id"], kept["s1_id"])) == {"S2-1": "A", "S2-2": "C"}
    kept0, info0 = resolve_one_owner(df, 0.5, near_tie_margin=0.0)  # exact tie is a near-tie -> dropped
    assert set(kept0["source_record_id"]) == {"S2-2"} and info0["n_near_tie_dropped_records"] == 1


def test_one_owner_empty_and_below_threshold_and_violation_detected():
    kept, info = resolve_one_owner(_scored([("A", "S2-1", 0.2)]), 0.5)
    assert len(kept) == 0 and info["n_thresholded"] == 0
    with pytest.raises(AssertionError):
        verify_one_owner(_scored([("A", "S2-1", 0.9), ("B", "S2-1", 0.9)]))


# ---------------------------------------------------------------- metrics
def test_entity_metrics_match_p3_reference_on_random_cases():
    rng = np.random.default_rng(0)
    ids = [f"S1-{i}" for i in range(300)]
    pool = [f"S2-{i}" for i in range(40)] + [f"S3-{i}" for i in range(40)]
    truth_rows, pred_rows, ref = [], [], []
    for s in ids:
        t = set(rng.choice(pool, size=rng.integers(0, 5), replace=False))
        p = set(rng.choice(pool, size=rng.integers(0, 5), replace=False))
        if t and rng.random() < 0.5:  # make overlap likely
            p |= set(list(t)[: rng.integers(0, len(t) + 1)])
        truth_rows += [(s, r) for r in t]
        pred_rows += [(s, r) for r in p]
        ref.append(compute_entity_f05(t, p))
    m = evaluate_entities(ids, pd.DataFrame(truth_rows, columns=["s1_id", "source_record_id"]),
                          pd.DataFrame(pred_rows, columns=["s1_id", "source_record_id"]))
    ref = np.array(ref)
    assert m["macro_f05"] == pytest.approx(ref[:, 0].mean())
    assert m["macro_precision"] == pytest.approx(ref[:, 1].mean())
    assert m["macro_recall"] == pytest.approx(ref[:, 2].mean())


def test_zero_candidate_entities_and_singletons_are_counted():
    truth = pd.DataFrame([("A", "S2-1")], columns=["s1_id", "source_record_id"])
    pred = pd.DataFrame([("B", "S2-9")], columns=["s1_id", "source_record_id"])
    m = evaluate_entities(["A", "B", "C"], truth, pred)  # A missed (0), B false match on singleton (0), C correct empty (1)
    assert m["macro_f05"] == pytest.approx(1 / 3) and m["n_singletons"] == 2 and m["pair_fp"] == 1 and m["pair_fn"] == 1


def test_source_segments_restrict_truth_and_predictions():
    truth = pd.DataFrame([("A", "S2-1"), ("A", "S3-1")], columns=["s1_id", "source_record_id"])
    pred = pd.DataFrame([("A", "S2-1")], columns=["s1_id", "source_record_id"])
    seg = evaluate_with_segments(["A"], truth, pred, {"A": "US"})
    assert seg["source"]["S2"]["macro_f05"] == pytest.approx(1.0)   # S2 perfectly matched
    assert seg["source"]["S3"]["macro_f05"] == pytest.approx(0.0)   # S3 match missed
    assert 0 < seg["overall"]["macro_f05"] < 1 and "US" in seg["country"]


def test_truth_pairs_from_lookup():
    t = truth_pairs({"A": {"S2-1", "S3-2"}, "B": set()}, ["A", "B", "C"])
    assert len(t) == 2 and set(t["s1_id"]) == {"A"}


# ---------------------------------------------------------------- split / thresholds
def test_grouped_split_is_disjoint_deterministic_and_complete():
    ids = [f"S1-{i}" for i in range(1000)]
    tr, va = grouped_split(ids, 0.2, seed=7)
    assert not (tr & va) and tr | va == set(ids) and len(va) == 200
    assert grouped_split(ids, 0.2, seed=7) == (tr, va) and grouped_split(ids, 0.2, seed=8)[1] != va


def test_threshold_grid_default_and_search_prefers_higher_threshold_on_ties():
    g = make_threshold_grid({"start": 0.30, "stop": 0.90, "step": 0.05})
    assert g[0] == 0.30 and g[-1] == 0.90 and len(g) == 13
    scored = _scored([("A", "S2-1", 0.95), ("B", "S2-2", 0.10)])
    truth = pd.DataFrame([("A", "S2-1")], columns=["s1_id", "source_record_id"])
    best, table = search_threshold(scored, ["A", "B"], truth, g, "macro_f05", one_owner_cfg={"near_tie_margin": 0.05})
    assert best["macro_f05"] == pytest.approx(1.0) and best["threshold"] == 0.90  # all thresholds tie -> highest
    assert len(table) == len(g)


# ---------------------------------------------------------------- writers / log
def test_write_and_verify_matching_results(tmp_path):
    m = _scored([("A", "S2-1", 0.9), ("A", "S3-1", 0.95)])
    p = write_matching_results(m, ["A", "B"], tmp_path / "m.tsv")
    lines = p.read_text().splitlines()
    assert lines == [MATCHING_HEADER, "A\tS3-1,S2-1", "B\t"]  # score-descending, empty list for no match
    assert verify_output_file(p, ["A", "B"], MATCHING_HEADER, check_one_owner=True) == {"rows": 2, "ids": 2}
    with pytest.raises(AssertionError):
        verify_output_file(p, ["A", "B", "C"], MATCHING_HEADER)


def test_experiment_log_append(tmp_path):
    p = tmp_path / "log.csv"
    append_experiment({"experiment_id": "x1", "owner": "P1", "K": 50, "validation_f05": 0.5}, p)
    append_experiment({"experiment_id": "x2"}, p)
    df = pd.read_csv(p, dtype=str, keep_default_na=False)
    assert list(df.columns) == COLUMNS and list(df["experiment_id"]) == ["x1", "x2"]
    with pytest.raises(KeyError):
        append_experiment({"bogus": 1}, p)
