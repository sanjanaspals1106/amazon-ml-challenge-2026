import sys, os, time, csv, json, random
import pandas as pd

DATA = "/mnt/user-data/uploads"
K = 50
N_SAMPLE = 10000
SEED = 42
# Sandbox is 1 CPU / ~3.9GB RAM. Loading the real S2+S3 pool (~10.3M rows) fully into
# pandas OOM-killed the process twice before any processing even started. This version
# streams every large file row-by-row (csv.reader) instead of pandas.read_csv, so memory
# never holds more than what's needed:
#   - every ground-truth match for the sampled S1 entities is guaranteed included in the
#     corpus (so recall is measured honestly, not inflated/deflated by missing truth)
#   - remaining corpus budget is filled by Bernoulli-sampling distractor rows while streaming
# This is a resource-constrained approximation of "full S2/S3 pool", not the literal full pool.
CORPUS_CAP = int(os.environ.get("CORPUS_CAP", "600000"))

random.seed(SEED)

def stream_tsv(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        for row in reader:
            if len(row) != len(header):
                continue
            yield dict(zip(header, row))

def main():
    print("Pass 1: streaming ground truth...", flush=True)
    id2matches = {}
    n_gt = 0
    for row in stream_tsv(os.path.join(DATA, "train_ground_truth.tsv")):
        n_gt += 1
        v = row["matched_entity_ids"]
        if v:
            id2matches[row["source1_entity_id"]] = v.split(",")
    print(f"  gt rows={n_gt:,}  with matches={len(id2matches):,}", flush=True)

    print("Pass 2: reservoir-sampling S1 entities with matches...", flush=True)
    reservoir = []
    seen = 0
    for row in stream_tsv(os.path.join(DATA, "train_source1.tsv")):
        eid = row["entity_id"]
        if eid not in id2matches:
            continue
        seen += 1
        item = row
        if len(reservoir) < N_SAMPLE:
            reservoir.append(item)
        else:
            j = random.randint(0, seen - 1)
            if j < N_SAMPLE:
                reservoir[j] = item
    print(f"  eligible S1 (has gt)={seen:,}  sampled={len(reservoir):,}", flush=True)

    sample = pd.DataFrame(reservoir)
    print(sample["country"].value_counts(), flush=True)

    gt_sub = {r["entity_id"]: set(id2matches[r["entity_id"]]) for r in reservoir}
    all_true_ids = set()
    for v in gt_sub.values():
        all_true_ids |= v
    print(f"  true-match entities to guarantee in corpus: {len(all_true_ids):,}", flush=True)
    id2matches = None  # free the big dict, no longer needed

    print("Pass 3: streaming S2 + S3, building bounded corpus...", flush=True)
    APPROX_TOTAL = 10_320_000
    p_keep = min(CORPUS_CAP / APPROX_TOTAL, 1.0)
    corpus_rows = []
    seen_ids = set()
    for path in [os.path.join(DATA, "train_source2_1.tsv"), os.path.join(DATA, "train_source3_1.tsv")]:
        for row in stream_tsv(path):
            eid = row["entity_id"]
            if eid in seen_ids:
                continue
            is_true = eid in all_true_ids
            if is_true or random.random() < p_keep:
                corpus_rows.append(row)
                seen_ids.add(eid)
    corpus = pd.DataFrame(corpus_rows)
    corpus_rows = None
    n_s2 = corpus["entity_id"].str.startswith("S2").sum()
    n_s3 = corpus["entity_id"].str.startswith("S3").sum()
    print(f"  corpus built: {len(corpus):,} records ({n_s2:,} S2, {n_s3:,} S3)", flush=True)
    print(corpus["country"].value_counts(), flush=True)
    missing_true = all_true_ids - seen_ids
    print(f"  true-match ids missing from corpus (should be 0): {len(missing_true)}", flush=True)

    s2 = corpus[corpus["entity_id"].str.startswith("S2")][["entity_id", "business_name", "business_address", "country"]].reset_index(drop=True)
    s3 = corpus[corpus["entity_id"].str.startswith("S3")][["entity_id", "business_name", "business_address", "country"]].reset_index(drop=True)
    s1_query = sample[["entity_id", "business_name", "business_address", "country"]].copy()
    country_map = dict(zip(sample["entity_id"], sample["country"]))

    results = {}
    for variant in ["baseline", "modified"]:
        sys.path.insert(0, f"/home/claude/eval/{variant}")
        for mod in list(sys.modules):
            if mod == "src" or mod.startswith("src."):
                del sys.modules[mod]
        from src.blocking.candidate_generator import CandidateGenerator

        print(f"\n=== {variant.upper()} ===", flush=True)
        gen = CandidateGenerator(top_k=K, verbose=True)

        t0 = time.time()
        gen.fit(source2_df=s2, source3_df=s3)
        fit_time = time.time() - t0

        t0 = time.time()
        cands = gen.generate_candidates(source1_df=s1_query)
        gen_time = time.time() - t0
        per_s1_ms = gen_time / len(s1_query) * 1000

        results[variant] = {"fit_time_s": fit_time, "gen_time_s": gen_time, "per_s1_ms": per_s1_ms, "cands": cands}
        print(f"fit_time={fit_time:.1f}s gen_time={gen_time:.1f}s per_s1={per_s1_ms:.2f}ms", flush=True)
        sys.path.remove(f"/home/claude/eval/{variant}")

    def compute_metrics(cands_df, tag):
        grouped = cands_df.groupby("s1_id")["source_record_id"].apply(set)
        recalled = total = 0
        us_recalled = us_total = 0
        in_recalled = in_total = 0
        s2_gt_recalled = s2_gt_total = 0
        s3_gt_recalled = s3_gt_total = 0
        for s1_id, truth in gt_sub.items():
            cand_set = grouped.get(s1_id, set())
            hit = len(truth & cand_set) > 0
            total += 1
            recalled += 1 if hit else 0
            country = country_map.get(s1_id)
            if country == "US":
                us_total += 1; us_recalled += 1 if hit else 0
            elif country == "India":
                in_total += 1; in_recalled += 1 if hit else 0
            for t in truth:
                if t.startswith("S2"):
                    s2_gt_total += 1
                    if t in cand_set: s2_gt_recalled += 1
                elif t.startswith("S3"):
                    s3_gt_total += 1
                    if t in cand_set: s3_gt_recalled += 1

        avg_cands = grouped.apply(len).mean() if len(grouped) else 0.0
        method_counts = cands_df["retrieval_method"].value_counts().to_dict()

        print(f"\n--- {tag} metrics (K={K}) ---")
        print(f"Candidate recall (>=1 true match in top-{K}): {recalled/total*100:.2f}%  ({recalled}/{total})")
        if us_total: print(f"  US:    {us_recalled/us_total*100:.2f}%  ({us_recalled}/{us_total})")
        if in_total: print(f"  India: {in_recalled/in_total*100:.2f}%  ({in_recalled}/{in_total})")
        if s2_gt_total: print(f"  S2 ground-truth match recall: {s2_gt_recalled/s2_gt_total*100:.2f}% ({s2_gt_recalled}/{s2_gt_total})")
        if s3_gt_total: print(f"  S3 ground-truth match recall: {s3_gt_recalled/s3_gt_total*100:.2f}% ({s3_gt_recalled}/{s3_gt_total})")
        print(f"Avg candidates/S1: {avg_cands:.2f}")
        print(f"Retrieval method breakdown: {method_counts}")
        return {
            "candidate_recall_pct": recalled/total*100,
            "us_recall_pct": (us_recalled/us_total*100) if us_total else None,
            "india_recall_pct": (in_recalled/in_total*100) if in_total else None,
            "s2_gt_recall_pct": (s2_gt_recalled/s2_gt_total*100) if s2_gt_total else None,
            "s3_gt_recall_pct": (s3_gt_recalled/s3_gt_total*100) if s3_gt_total else None,
            "avg_candidates_per_s1": float(avg_cands),
            "method_counts": {k: int(v) for k, v in method_counts.items()},
        }

    summary = {}
    for variant in ["baseline", "modified"]:
        m = compute_metrics(results[variant]["cands"], variant)
        m["fit_time_s"] = results[variant]["fit_time_s"]
        m["gen_time_s"] = results[variant]["gen_time_s"]
        m["per_s1_ms"] = results[variant]["per_s1_ms"]
        summary[variant] = m

    with open("/home/claude/eval/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved summary.json", flush=True)

if __name__ == "__main__":
    main()
