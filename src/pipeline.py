"""
Baseline pipeline orchestrator (P1 / integration).

    python -m src.pipeline --config configs/baseline.yaml                  # development run on a train subset
    python -m src.pipeline --config configs/baseline.yaml --mode test ...  # inference (guarded, see below)

Stages (each implemented in its owner's module; this file only wires them together):

    load (src.data_loading)
      -> candidates  (P2: src.blocking.CandidateGenerator via src.candidates)
      -> features    (P3: src.features.build_features)
      -> labels      (P3: src.model.build_labels, from the real ground truth; negatives = retrieved non-matches)
      -> grouped split by S1 entity (src.postprocessing.grouped_split)
      -> matcher     (P3: src.model.train_matcher / predict_match_scores)
      -> threshold search + one-owner rule + entity-level metrics (src.postprocessing)
      -> outputs (candidate_pairs.tsv, candidates_detailed.tsv, matching_results.tsv, baseline_metrics.json)
      -> experiment log row (experiments/experiment_log.csv)
"""

import argparse
import hashlib
import inspect
import json
import logging
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import yaml

import src.features.normalization as _p3_norm
import src.features.pair_features as _p3_pair_features
from src.candidates import load_or_generate_candidates
from src.data_loading import (
    DataError, dataset_paths, load_split, resolve_path, restrict_pool, sample_s1, validate_candidates,
)
from src.experiment_log import append_experiment
from src.features import build_features
from src.model import (
    EntityResolutionMatcher, build_ground_truth_lookup, build_labels, predict_match_scores, train_matcher,
)
from src.postprocessing import (
    apply_decision_rule, candidate_recall, evaluate_with_segments, grouped_split, make_threshold_grid,
    search_threshold, truth_pairs, verify_one_owner,
)
from src.postprocessing.output import (
    CANDIDATE_PAIRS_HEADER, MATCHING_HEADER, verify_output_file, write_candidate_pairs,
    write_candidates_detailed, write_matching_results,
)

logger = logging.getLogger("pipeline")


# ----------------------------------------------------------------------------------------------- helpers
def load_config(path: str) -> dict:
    p = resolve_path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for section in ("paths", "candidates", "model", "validation", "threshold_search", "outputs"):
        if section not in cfg:
            raise DataError(f"Config {p} is missing required section '{section}'")
    return cfg


def _clean(o):
    """Make objects JSON-safe (numpy -> python, NaN -> None)."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if hasattr(o, "item") and not isinstance(o, (str, bytes)):
        try:
            o = o.item()
        except Exception:
            pass
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return None
    return o


def _source_hash(*modules) -> str:
    h = hashlib.sha1()
    for m in modules:
        h.update(Path(inspect.getsourcefile(m)).read_bytes())
    return h.hexdigest()


def _needed_source_frames(cand: pd.DataFrame, s2: pd.DataFrame, s3: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Only the S2/S3 records referenced by candidates (P3's build_features precomputes every record it is given)."""
    need = set(cand["source_record_id"].unique())
    return {"S2": s2[s2["entity_id"].isin(need)], "S3": s3[s3["entity_id"].isin(need)]}


def _features_with_cache(cand, s1_sample, s2, s3, cfg, cand_key, use_cache) -> pd.DataFrame:
    """P3's build_features(candidate_df, s1_df, source_df, config), cached on disk (git-ignored)."""
    cache_dir = resolve_path(cfg["paths"].get("cache_dir", "outputs/cache"))
    fkey = hashlib.sha1(json.dumps({
        "cand": cand_key, "features_cfg": cfg.get("features", {}),
        "p3": _source_hash(_p3_pair_features, _p3_norm),
    }, sort_keys=True).encode()).hexdigest()[:12]
    path = cache_dir / f"features_{fkey}.parquet"
    if use_cache and path.exists():
        feat = pd.read_parquet(path)
        logger.info("Feature cache HIT: %s (%s rows)", path.name, f"{len(feat):,}")
        return feat
    src_frames = _needed_source_frames(cand, s2, s3)
    logger.info("Building features for %s pairs (%s S1, %s S2, %s S3 records)...", f"{len(cand):,}",
                f"{len(s1_sample):,}", f"{len(src_frames['S2']):,}", f"{len(src_frames['S3']):,}")
    feat = build_features(cand, s1_sample, src_frames, config=cfg.get("features", {}))
    if len(feat) != len(cand):
        raise DataError(f"Feature/candidate interface mismatch: build_features returned {len(feat)} rows for {len(cand)} candidates")
    if not ((feat["s1_id"].values == cand["s1_id"].values).all() and (feat["source_record_id"].values == cand["source_record_id"].values).all()):
        raise DataError("Feature/candidate interface mismatch: build_features changed the candidate row order/IDs")
    feat["country"] = cand["country"].values
    if use_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)
        feat.to_parquet(path, index=False)
    return feat


def _check_sample_guard(n: Optional[int], total: int, limit: int, allowed: bool, what: str) -> None:
    """Refuse accidental full-scale runs: a run over more than `limit` S1 entities needs an explicit flag."""
    size = total if (n is None or n <= 0 or n >= total) else n
    if size > limit and not allowed:
        raise SystemExit(
            f"Refusing to run {what} on {size:,} S1 entities (limit {limit:,} without the explicit flag). "
            f"Set a smaller sample (dev.sample_s1 / test.sample_s1 / --sample-s1) or pass --allow-full / --allow-full-test.")


# ----------------------------------------------------------------------------------------------- train
def run_train(cfg: dict, args) -> dict:
    t_all = time.time()
    timings: Dict[str, float] = {}
    dev = cfg.get("dev", {})
    n_s1 = args.sample_s1 if args.sample_s1 is not None else dev.get("sample_s1")
    seed = int(dev.get("sample_seed", 42))
    use_cache = bool(cfg["candidates"].get("use_cache", True)) and not args.no_cache
    if args.workers is not None:
        cfg["candidates"]["workers"] = args.workers

    t = time.time()
    data = load_split(cfg, "train")
    s1_all, s2, s3, gt = data["s1"], data["s2"], data["s3"], data["gt"]
    _check_sample_guard(n_s1, len(s1_all), int(dev.get("max_s1_without_flag", 200000)), args.allow_full, "train mode")
    s1 = sample_s1(s1_all, n_s1, seed)
    s1_ids = s1["entity_id"].tolist()
    logger.info("Development sample: %s S1 entities (seed=%d)", f"{len(s1):,}", seed)

    # Ground truth for the sampled S1 (P3's lookup); every referenced record must exist in the pool.
    gt_s = gt[gt["source1_entity_id"].isin(set(s1_ids))]
    gt_lookup = build_ground_truth_lookup(gt_s)
    all_true_ids = {i for v in gt_lookup.values() for i in v}
    known = set(s2["entity_id"]) | set(s3["entity_id"])
    missing = all_true_ids - known
    if missing:
        raise DataError(f"Ground truth references {len(missing)} S2/S3 IDs missing from the source files (e.g. {sorted(missing)[:3]})")

    pool_frac = float(dev.get("pool_sample_frac", 1.0))
    s2p, s3p = restrict_pool(s2, s3, pool_frac, seed, keep_ids=all_true_ids)
    pool_desc = {"frac": pool_frac, "seed": seed, "n_s2": len(s2p), "n_s3": len(s3p)}
    timings["load_s"] = round(time.time() - t, 1)

    # ---- candidates (P2)
    t = time.time()
    paths = dataset_paths(cfg, "train")
    cand, cinfo = load_or_generate_candidates(
        s1, {"S2": s2p, "S3": s3p}, cfg, "train", [paths["source1"], paths["source2"], paths["source3"]], pool_desc, use_cache)
    vinfo = validate_candidates(cand, expected_s1_ids=s1_ids)
    timings["candidates_s"] = round(time.time() - t, 1)
    logger.info("Candidates: %s pairs for %s S1 (%s S1 without candidates) [cache_hit=%s]",
                f"{vinfo['n_pairs']:,}", f"{vinfo['n_s1_with_candidates']:,}", vinfo["n_s1_without_candidates"], cinfo["cache_hit"])

    # ---- features (P3) + labels (P3)
    t = time.time()
    feat = _features_with_cache(cand, s1, s2p, s3p, cfg, cinfo["cache_key"], use_cache)
    feat["label"] = build_labels(cand, gt_lookup)
    timings["features_labels_s"] = round(time.time() - t, 1)
    n_pos, n_neg = int((feat["label"] == 1).sum()), int((feat["label"] == 0).sum())
    logger.info("Labels: %s positives / %s negatives (retrieved non-matches)", f"{n_pos:,}", f"{n_neg:,}")
    if n_pos == 0 or n_neg == 0:
        raise DataError(f"Training data has a single class (positives={n_pos}, negatives={n_neg}); cannot train")

    # ---- grouped split by S1 entity
    vcfg = cfg["validation"]
    train_ids, val_ids = grouped_split(s1_ids, float(vcfg.get("val_fraction", 0.2)), int(vcfg.get("random_state", 42)))
    is_val = feat["s1_id"].isin(val_ids)
    train_df, val_df = feat[~is_val].reset_index(drop=True), feat[is_val].reset_index(drop=True)
    assert not (set(train_df["s1_id"]) & set(val_df["s1_id"])), "S1 leakage between train and validation"
    logger.info("Grouped split: %s train S1 / %s val S1 entities (%s / %s pairs)", f"{len(train_ids):,}", f"{len(val_ids):,}",
                f"{len(train_df):,}", f"{len(val_df):,}")

    # ---- model (P3)
    t = time.time()
    model = train_matcher(train_df, train_df["label"].values, cfg["model"])
    val_scores = predict_match_scores(model, val_df)
    timings["train_predict_s"] = round(time.time() - t, 1)
    if len(val_scores) != len(val_df) or not ((val_scores >= 0) & (val_scores <= 1)).all():
        raise DataError("Matcher returned invalid probabilities (wrong length or outside [0,1])")
    scored = val_df[["s1_id", "source_record_id", "source", "country"]].copy()
    scored["score"] = val_scores

    # ---- threshold search with the real decision rule (P1)
    t = time.time()
    val_list = sorted(val_ids)
    truth_val = truth_pairs(gt_lookup, val_list)
    country_by_s1 = dict(zip(s1["entity_id"], s1["country"]))
    oo_cfg = cfg.get("one_owner", {})
    oo = oo_cfg if (oo_cfg.get("enabled", True) and cfg["threshold_search"].get("apply_one_owner", True)) else None
    grid = make_threshold_grid(cfg["threshold_search"].get("grid"))
    best, table = search_threshold(scored, val_list, truth_val, grid, cfg["threshold_search"].get("metric", "macro_f05"), oo)
    thr = float(best["threshold"])
    logger.info("Threshold grid:\n%s", table[["threshold", "macro_f05", "macro_precision", "macro_recall", "singleton_f05",
                                            "non_singleton_f05", "n_predicted_pairs", "macro_f05_before_one_owner"]].round(4).to_string(index=False))
    logger.info("Selected threshold %.2f (%s=%.4f)", thr, cfg["threshold_search"].get("metric", "macro_f05"), best[cfg["threshold_search"].get("metric", "macro_f05")])

    # ---- final decision on validation entities + segments
    acc, oo_info = apply_decision_rule(scored, thr, oo_cfg if oo_cfg.get("enabled", True) else None)
    if oo_cfg.get("enabled", True):
        verify_one_owner(acc)
    seg = evaluate_with_segments(val_list, truth_val, acc, country_by_s1)
    seg_pre = evaluate_with_segments(val_list, truth_val, scored[scored["score"] >= thr], country_by_s1)["overall"]
    rec = candidate_recall(truth_val, cand[cand["s1_id"].isin(val_ids)])
    timings["threshold_eval_s"] = round(time.time() - t, 1)

    # ---- outputs
    t = time.time()
    outs = {k: resolve_path(v) for k, v in cfg["outputs"].items()}
    write_candidate_pairs(cand, s1_ids, outs["candidate_pairs"])
    write_candidates_detailed(cand, outs["candidates_detailed"])
    write_matching_results(acc, val_list, outs["matching_results"])
    chk = {
        "candidate_pairs": verify_output_file(outs["candidate_pairs"], s1_ids, CANDIDATE_PAIRS_HEADER),
        "matching_results": verify_output_file(outs["matching_results"], val_list, MATCHING_HEADER, check_one_owner=True),
    }
    model_path = resolve_path(cfg["paths"]["model_path"])
    model.save(str(model_path))
    timings["outputs_s"] = round(time.time() - t, 1)
    timings["total_s"] = round(time.time() - t_all, 1)

    o = seg["overall"]
    countries = {c: v["macro_f05"] for c, v in seg["country"].items()}
    metrics = {
        "mode": "train", "date": datetime.now().isoformat(timespec="seconds"),
        "validation_f05": o["macro_f05"], "precision": o["macro_precision"], "recall": o["macro_recall"],
        "US_f05": countries.get("US"), "India_f05": countries.get("India"),
        "S2_f05": seg["source"]["S2"]["macro_f05"], "S3_f05": seg["source"]["S3"]["macro_f05"],
        "threshold": thr, "n_s1_evaluated": o["n_entities"],
        "n_candidate_pairs": int(len(cand)), "n_predicted_matches": int(len(acc)),
        "details": {
            "metric_definition": "entity-level macro average over ALL validation S1 entities incl. singletons and zero-candidate entities",
            "country_f05": countries, "segments": seg, "before_one_owner_overall": seg_pre,
            "one_owner": {"enabled": bool(oo_cfg.get("enabled", True)), "near_tie_margin": oo_cfg.get("near_tie_margin"), **oo_info,
                          "violations": 0},
            "candidate_recall_validation": rec,
            "threshold_table": table.to_dict(orient="records"),
            "sample": {"n_s1_sampled": len(s1), "seed": seed, "n_train_s1": len(train_ids), "n_val_s1": len(val_ids),
                       "pool_sample_frac": pool_frac, "n_s2_pool": len(s2p), "n_s3_pool": len(s3p),
                       "n_s1_without_candidates": vinfo["n_s1_without_candidates"]},
            "candidates": {"top_k": cfg["candidates"].get("top_k"), "n_pairs": int(len(cand)), **cinfo},
            "labels": {"n_positive": n_pos, "n_negative": n_neg, "positive_rate": n_pos / (n_pos + n_neg),
                       "n_train_pairs": int(len(train_df)), "n_val_pairs": int(len(val_df))},
            "model": {"configured": cfg["model"].get("model_type"), "effective": model.model_type, "n_features": len(model.feature_names_),
                      "path": str(model_path)},
            "output_checks": chk, "timings_seconds": timings,
        },
    }
    metrics_path = resolve_path(cfg["outputs"]["metrics"])
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(_clean(metrics), indent=2), encoding="utf-8")

    if cfg.get("experiment", {}).get("log_run", True) and not args.no_log:
        pool_note = "full pool" if pool_frac >= 1.0 else f"REDUCED pool frac={pool_frac} (wiring only)"
        append_experiment({
            "experiment_id": f"{cfg['experiment'].get('name', 'run')}_{datetime.now():%Y%m%d_%H%M%S}",
            "date": datetime.now().strftime("%Y-%m-%d"), "owner": cfg["experiment"].get("owner", "P1"),
            "blocking": "P2 CandidateGenerator (exact + IDF addr/name + name-p4, per-country)", "K": cfg["candidates"].get("top_k"),
            "normalization": "P2 src/preprocessing/normalizer + P3 src/features/normalization",
            "features": f"P3 build_features ({metrics['details']['model']['n_features']} numeric)",
            "model": f"{model.model_type} (configured {cfg['model'].get('model_type')})",
            "threshold": thr, "validation_f05": round(o["macro_f05"], 4), "precision": round(o["macro_precision"], 4),
            "recall": round(o["macro_recall"], 4),
            "US_f05": None if countries.get("US") is None else round(countries["US"], 4),
            "India_f05": None if countries.get("India") is None else round(countries["India"], 4),
            "S2_f05": round(metrics["S2_f05"], 4), "S3_f05": round(metrics["S3_f05"], 4),
            "notes": (cfg["experiment"].get("notes") or "") + f" dev sample={len(s1)} S1 seed={seed}, val={len(val_ids)} S1, {pool_note}, "
                     f"one_owner margin={oo_cfg.get('near_tie_margin')}; {args.notes or ''}".strip(),
        }, resolve_path(cfg["paths"]["experiment_log"]))
    logger.info("Metrics written to %s", metrics_path)
    return metrics


# ----------------------------------------------------------------------------------------------- test
def run_test(cfg: dict, args) -> dict:
    """Inference on the test split. Guarded: without an explicit sample it needs --allow-full-test."""
    t_all = time.time()
    tcfg = cfg.get("test", {})
    n_s1 = args.sample_s1 if args.sample_s1 is not None else tcfg.get("sample_s1")
    seed = int(tcfg.get("sample_seed", 42))
    use_cache = bool(cfg["candidates"].get("use_cache", True)) and not args.no_cache
    if args.workers is not None:
        cfg["candidates"]["workers"] = args.workers

    model_path = resolve_path(cfg["paths"]["model_path"])
    if not model_path.exists():
        raise FileNotFoundError(f"Trained model not found: {model_path}. Run --mode train first.")
    thr = args.threshold if args.threshold is not None else tcfg.get("threshold")
    if thr is None:
        mp = resolve_path(cfg["outputs"]["metrics"])
        if not mp.exists():
            raise FileNotFoundError(f"No threshold given and {mp} not found. Run --mode train first or pass --threshold.")
        thr = json.loads(mp.read_text())["threshold"]
    thr = float(thr)

    data = load_split(cfg, "test")
    s1_all, s2, s3 = data["s1"], data["s2"], data["s3"]
    _check_sample_guard(n_s1, len(s1_all), int(tcfg.get("max_s1_without_flag", 100000)), args.allow_full_test, "TEST mode")
    s1 = sample_s1(s1_all, n_s1, seed)
    s1_ids = s1["entity_id"].tolist()
    pool_frac = float(tcfg.get("pool_sample_frac", 1.0))
    s2p, s3p = restrict_pool(s2, s3, pool_frac, seed)
    pool_desc = {"frac": pool_frac, "seed": seed, "n_s2": len(s2p), "n_s3": len(s3p)}
    paths = dataset_paths(cfg, "test")
    cand, cinfo = load_or_generate_candidates(
        s1, {"S2": s2p, "S3": s3p}, cfg, "test", [paths["source1"], paths["source2"], paths["source3"]], pool_desc, use_cache)
    vinfo = validate_candidates(cand, expected_s1_ids=s1_ids)
    feat = _features_with_cache(cand, s1, s2p, s3p, cfg, cinfo["cache_key"], use_cache)

    model = EntityResolutionMatcher.load(str(model_path))
    missing = [c for c in model.feature_names_ if c not in feat.columns]
    if missing:
        raise DataError(f"Feature/model interface mismatch: features lack columns the model was trained on: {missing[:5]}")
    scored = feat[["s1_id", "source_record_id", "source", "country"]].copy()
    scored["score"] = predict_match_scores(model, feat)
    oo_cfg = cfg.get("one_owner", {})
    acc, oo_info = apply_decision_rule(scored, thr, oo_cfg if oo_cfg.get("enabled", True) else None)
    if oo_cfg.get("enabled", True):
        verify_one_owner(acc)

    outs = {k: resolve_path(v) for k, v in cfg["outputs_test"].items()}
    write_candidate_pairs(cand, s1_ids, outs["candidate_pairs"])
    write_candidates_detailed(cand, outs["candidates_detailed"])
    write_matching_results(acc, s1_ids, outs["matching_results"])
    verify_output_file(outs["candidate_pairs"], s1_ids, CANDIDATE_PAIRS_HEADER)
    verify_output_file(outs["matching_results"], s1_ids, MATCHING_HEADER, check_one_owner=True)
    summary = {"mode": "test", "threshold": thr, "n_s1": len(s1), "n_candidate_pairs": int(len(cand)),
               "n_predicted_matches": int(len(acc)), "one_owner": oo_info, "seconds": round(time.time() - t_all, 1),
               "outputs": {k: str(v) for k, v in outs.items()}}
    logger.info("Test-mode summary: %s", json.dumps(_clean(summary)))
    return summary


# ----------------------------------------------------------------------------------------------- CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description="Baseline entity-resolution pipeline (P1 integration)")
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--mode", choices=["train", "test"], default="train")
    ap.add_argument("--sample-s1", type=int, default=None, help="Override dev/test sample size (number of S1 entities)")
    ap.add_argument("--workers", type=int, default=None, help="Override candidates.workers")
    ap.add_argument("--no-cache", action="store_true", help="Ignore and do not write candidate/feature caches")
    ap.add_argument("--no-log", action="store_true", help="Do not append a row to the experiment log")
    ap.add_argument("--notes", default="", help="Free-text note stored in the experiment log")
    ap.add_argument("--threshold", type=float, default=None, help="(test mode) override the selected threshold")
    ap.add_argument("--allow-full", action="store_true", help="Allow train mode on more than dev.max_s1_without_flag S1 entities")
    ap.add_argument("--allow-full-test", action="store_true", help="Allow test mode on ALL test S1 entities (1.73M; hours)")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", stream=sys.stdout)
    cfg = load_config(args.config)
    result = run_train(cfg, args) if args.mode == "train" else run_test(cfg, args)
    if args.mode == "train":
        print("\n" + "=" * 70)
        print(f" VALIDATION (grouped by S1, n={result['n_s1_evaluated']:,} entities)  threshold={result['threshold']:.2f}")
        print(f"   macro F0.5={result['validation_f05']:.4f}  precision={result['precision']:.4f}  recall={result['recall']:.4f}")
        print(f"   US={result['US_f05']}  India={result['India_f05']}  S2={result['S2_f05']:.4f}  S3={result['S3_f05']:.4f}")
        print("=" * 70)
    return result


if __name__ == "__main__":
    main()
