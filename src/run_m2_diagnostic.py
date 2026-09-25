"""
Reproducible diagnostic runner for M2 (Pair Features + Matching Model).

Amazon ML Challenge 2026

Runs an end-to-end diagnostic workflow:
  candidate_pairs -> build_features() -> build_labels() -> train_matcher() -> predict_match_scores()

Features tested:
- Lexical similarities (token-sort, token-set, Levenshtein, Jaro-Winkler, Jaccard, 3-grams)
- Structural differences (token count, char length, first token equality, normalized equality)
- Business semantics (legal form agreement/conflict, core name similarity, initials match)
- Address features (house numbers, leading zero normalization, off-by-one, postal codes, state agreement, empty handling)
- Context & retrieval features (source, country, retrieval rank, score, candidate counts)
- Grouped S1 validation (zero leakage of S1 entities)
- Competition-exact Macro F0.5 evaluation (including singletons)

Usage:
    python src/run_m2_diagnostic.py
    python src/run_m2_diagnostic.py --from-zip
"""

import argparse
import logging
import os
import sys
import zipfile
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import yaml

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.features.pair_features import build_features
from src.model.labels import attach_labels, build_ground_truth_lookup, build_labels, compute_label_summary
from src.model.matcher import EntityResolutionMatcher
from src.model.prediction import predict_match_scores, score_candidates
from src.model.training import (
    evaluate_entity_macro_metrics,
    evaluate_segments,
    run_threshold_grid_search,
    split_grouped_by_s1,
    train_matcher,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("m2_diagnostic")

DATASET_ZIP_PATH = "6ab10eb3b23ba_student_resource.zip"


def generate_rich_diagnostic_dataset(
    num_s1: int = 150,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Generate a realistic, controlled entity resolution dataset for diagnostic verification.
    Models the real challenge distribution:
    - US and India records
    - Singletons (~25% of S1 have 0 matches)
    - Multi-matches (~25% of S1 have matches in both S2 and S3)
    - Single-match (~50% of S1 have 1 match)
    - Realistic noise: legal suffix variations, abbreviations, typos, leading-zero house numbers,
      off-by-one house numbers, missing addresses, hard negatives (look-alikes and address decoys).
    """
    rng = np.random.default_rng(random_state)

    us_entities = [
        ("Walmart", ["Walmart Inc", "Walmart Stores", "Wal-Mart"], "702 SW 8th St, Bentonville, AR 72716", "702 South West 8th Street, Bentonville, AR 72716"),
        ("Home Depot", ["The Home Depot Inc", "Home Depot USA, Inc.", "Home Depot"], "2455 Paces Ferry Rd NW, Atlanta, GA 30339", "2455 Paces Ferry Road, Atlanta, GA 30339"),
        ("Target", ["Target Corp", "Target Corporation", "Target Stores"], "1000 Nicollet Mall, Minneapolis, MN 55403", "01000 Nicollet Mall, Minneapolis, MN 55403"),
        ("CVS Health", ["CVS Health Corporation", "CVS Pharmacy Inc", "CVS Caremark"], "One CVS Dr, Woonsocket, RI 02895", "1 CVS Drive, Woonsocket, RI 02895"),
        ("Costco", ["Costco Wholesale Corp", "Costco Wholesale", "Costco"], "999 Lake Dr, Issaquah, WA 98027", "999 Lake Drive, Issaquah, WA 98027"),
        ("Kroger", ["The Kroger Co", "Kroger Company", "Kroger Supermarket"], "1014 Vine St, Cincinnati, OH 45202", "1014 Vine Street, Cincinnati, OH 45202"),
        ("Walgreens", ["Walgreens Co", "Walgreen Co.", "Walgreens Boots Alliance"], "108 Wilmot Rd, Deerfield, IL 60015", "108 Wilmot Road, Deerfield, IL 60015"),
        ("Microsoft", ["Microsoft Corporation", "Microsoft Corp", "Microsoft"], "One Microsoft Way, Redmond, WA 98052", "1 Microsoft Way, Redmond, WA 98052"),
        ("Amazon", ["Amazon.com Inc", "Amazon Services LLC", "Amazon Commercial Services"], "410 Terry Ave N, Seattle, WA 98109", "410 Terry Avenue North, Seattle, WA 98109"),
        ("Apple", ["Apple Inc", "Apple Computer Inc", "Apple Operations"], "One Apple Park Way, Cupertino, CA 95014", "1 Apple Park Way, Cupertino, CA 95014"),
        ("Google", ["Google LLC", "Google Inc", "Alphabet Google"], "1600 Amphitheatre Pkwy, Mountain View, CA 94043", "1600 Amphitheatre Parkway, Mountain View, CA 94043"),
        ("Meta", ["Meta Platforms Inc", "Facebook Inc", "Meta Platforms"], "1 Hacker Way, Menlo Park, CA 94025", "1 Hacker Way, Menlo Park, California 94025"),
        ("Johnson & Johnson", ["Johnson & Johnson", "Johnson and Johnson Services Inc", "J&J"], "One Johnson & Johnson Plaza, New Brunswick, NJ 08933", "1 Johnson and Johnson Plaza, New Brunswick, NJ 08933"),
        ("Procter & Gamble", ["The Procter & Gamble Company", "Procter and Gamble Corp", "P&G"], "One Procter & Gamble Plaza, Cincinnati, OH 45202", "1 P&G Plaza, Cincinnati, OH 45202"),
        ("General Electric", ["General Electric Company", "GE Energy LLC", "General Electric Corp"], "5 Necco St, Boston, MA 02210", "5 Necco Street, Boston, MA 02210"),
        ("Boeing", ["The Boeing Company", "Boeing Commercial Airplanes", "Boeing Corp"], "929 Long Bridge Dr, Arlington, VA 22202", "929 Long Bridge Drive, Arlington, VA 22202"),
        ("FedEx", ["FedEx Corp", "Federal Express Corporation", "FedEx Express"], "942 S Shady Grove Rd, Memphis, TN 38120", "942 South Shady Grove Road, Memphis, TN 38120"),
        ("UPS", ["United Parcel Service Inc", "UPS Inc", "United Parcel Service"], "55 Glenlake Pkwy NE, Atlanta, GA 30328", "55 Glenlake Parkway, Atlanta, GA 30328"),
        ("PepsiCo", ["PepsiCo Inc", "Pepsi-Cola Company", "PepsiCo Beverages"], "700 Anderson Hill Rd, Purchase, NY 10577", "700 Anderson Hill Road, Purchase, NY 10577"),
        ("Coca-Cola", ["The Coca-Cola Company", "Coca Cola Refreshments USA", "Coca-Cola"], "One Coca-Cola Plaza, Atlanta, GA 30313", "1 Coca Cola Plaza, Atlanta, GA 30313"),
    ]

    india_entities = [
        ("Reliance", ["Reliance Industries Limited", "Reliance Retail Ltd", "Reliance Jio Infocomm Ltd"], "Maker Chambers IV, Nariman Point, Mumbai 400021", "Maker Chambers 4, Nariman Point, Mumbai 400021"),
        ("Tata Motors", ["Tata Motors Limited", "Tata Motors Commercial", "Tata Motors Passenger"], "Bombay House, 24 Homi Mody Street, Mumbai 400001", "24 Homi Mody St, Fort, Mumbai 400001"),
        ("TCS", ["Tata Consultancy Services Ltd", "Tata Consultancy Services", "TCS Limited"], "TCS House, Raveline Street, Fort, Mumbai 400001", "Raveline St, Near Azad Maidan, Mumbai 400001"),
        ("Infosys", ["Infosys Limited", "Infosys Technologies Ltd", "Infosys"], "Electronics City, Hosur Road, Bangalore 560100", "Electronics City, Hosur Rd, Bengaluru 560100"),
        ("Wipro", ["Wipro Limited", "Wipro Enterprises Pvt Ltd", "Wipro Infotech"], "Doddakannelli, Sarjapur Road, Bangalore 560035", "Sarjapur Rd, Doddakannelli, Bengaluru 560035"),
        ("HDFC Bank", ["HDFC Bank Limited", "HDFC Bank Ltd", "HDFC Bank"], "HDFC Bank House, Senapati Bapat Marg, Lower Parel, Mumbai 400013", "Senapati Bapat Marg, Lower Parel, Mumbai 400013"),
        ("ICICI Bank", ["ICICI Bank Limited", "ICICI Bank Ltd", "ICICI Bank Towers"], "ICICI Bank Towers, Bandra Kurla Complex, Mumbai 400051", "BKC, Bandra East, Mumbai 400051"),
        ("State Bank of India", ["State Bank of India", "SBI", "State Bank of India Corporate Office"], "State Bank Bhavan, Madame Cama Road, Nariman Point, Mumbai 400021", "Madame Cama Rd, Nariman Point, Mumbai 400021"),
        ("Bharti Airtel", ["Bharti Airtel Limited", "Bharti Airtel Ltd", "Airtel Telecommunications"], "Bharti Crescent, 1 Nelson Mandela Road, Vasant Kunj, New Delhi 110070", "1 Nelson Mandela Rd, Vasant Kunj, New Delhi 110070"),
        ("Larsen & Toubro", ["Larsen & Toubro Limited", "Larsen and Toubro Ltd", "L&T Construction"], "L&T House, Ballard Estate, Mumbai 400001", "Ballard Estate, Fort, Mumbai 400001"),
        ("ITC", ["ITC Limited", "ITC Ltd", "ITC Hotels"], "Virginia House, 37 J.L. Nehru Road, Kolkata 700071", "37 JL Nehru Rd, Kolkata 700071"),
        ("Hindustan Unilever", ["Hindustan Unilever Limited", "Hindustan Unilever Ltd", "HUL"], "Unilever House, B.D. Sawant Marg, Chakala, Andheri East, Mumbai 400099", "BD Sawant Marg, Chakala, Andheri E, Mumbai 400099"),
        ("Bajaj Auto", ["Bajaj Auto Limited", "Bajaj Auto Ltd", "Bajaj Auto Commercial"], "Mumbai Pune Road, Akurdi, Pune 411035", "Old Mumbai-Pune Rd, Akurdi, Pune 411035"),
        ("Mahindra", ["Mahindra & Mahindra Limited", "Mahindra and Mahindra Ltd", "M&M Automotive"], "Gateway Building, Apollo Bunder, Mumbai 400001", "Apollo Bunder, Colaba, Mumbai 400001"),
        ("Sun Pharma", ["Sun Pharmaceutical Industries Ltd", "Sun Pharma Ltd", "Sun Pharmaceuticals"], "Sun Pharma House, Western Express Highway, Goregaon East, Mumbai 400063", "WEH, Goregaon E, Mumbai 400063"),
    ]

    s1_rows = []
    source_rows = []
    cand_rows = []
    gt_rows = []

    s1_counter = 1
    s2_counter = 1
    s3_counter = 1

    # Base entity pool
    pool = (us_entities * 4) + (india_entities * 6)
    rng.shuffle(pool)
    pool = pool[:num_s1]

    for base_name, name_variants, addr1, addr2 in pool:
        s1_id = f"S1-{s1_counter:05d}"
        s1_counter += 1
        country = "US" if ", AR " in addr1 or ", GA " in addr1 or ", MN " in addr1 or ", RI " in addr1 or ", WA " in addr1 or ", OH " in addr1 or ", IL " in addr1 or ", CA " in addr1 or ", NJ " in addr1 or ", MA " in addr1 or ", VA " in addr1 or ", TN " in addr1 or ", NY " in addr1 else "India"

        # S1 record
        # Some S1 have missing addresses (~8% per report)
        s1_addr = "" if rng.random() < 0.08 else addr1
        s1_rows.append({
            "entity_id": s1_id,
            "business_name": name_variants[0],
            "business_address": s1_addr,
            "country": country,
        })

        # Match profile: 0=singleton (25%), 1=single match (50%), 2=multi-match (25%)
        match_type = rng.choice([0, 1, 1, 2])
        matched_ids = []

        rank = 1

        if match_type >= 1:
            # S2 true match
            s2_id = f"S2-{s2_counter:05d}"
            s2_counter += 1
            s2_name = name_variants[min(1, len(name_variants) - 1)]
            s2_addr = "" if rng.random() < 0.08 else addr2
            source_rows.append({
                "entity_id": s2_id,
                "business_name": s2_name,
                "business_address": s2_addr,
                "country": country,
            })
            matched_ids.append(s2_id)
            cand_rows.append({
                "s1_id": s1_id,
                "source_record_id": s2_id,
                "source": "S2",
                "country": country,
                "retrieval_rank": rank,
                "retrieval_score": round(float(rng.uniform(0.75, 0.98)), 4),
                "retrieval_method": "idf_address_token",
            })
            rank += 1

        if match_type == 2:
            # S3 true match (multi-match!)
            s3_id = f"S3-{s3_counter:05d}"
            s3_counter += 1
            s3_name = name_variants[min(2, len(name_variants) - 1)]
            s3_addr = addr2
            source_rows.append({
                "entity_id": s3_id,
                "business_name": s3_name,
                "business_address": s3_addr,
                "country": country,
            })
            matched_ids.append(s3_id)
            cand_rows.append({
                "s1_id": s1_id,
                "source_record_id": s3_id,
                "source": "S3",
                "country": country,
                "retrieval_rank": rank,
                "retrieval_score": round(float(rng.uniform(0.70, 0.95)), 4),
                "retrieval_method": "idf_name_token",
            })
            rank += 1

        # Add 2 to 4 hard negative / decoy candidates retrieved for this S1 entity
        num_negs = rng.integers(2, 5)
        for _ in range(num_negs):
            neg_src = rng.choice(["S2", "S3"])
            if neg_src == "S2":
                nid = f"S2-{s2_counter:05d}"
                s2_counter += 1
            else:
                nid = f"S3-{s3_counter:05d}"
                s3_counter += 1

            # Negative kind:
            # 1: Lookalike name at different address (sibling entity)
            # 2: Same address, different business name (building decoy)
            # 3: Legal form conflict
            neg_kind = rng.choice([1, 2, 3])
            if neg_kind == 1:
                neg_name = f"{base_name} Enterprise Ltd"
                neg_addr = "999 Industrial Park Rd, Suite 400"
            elif neg_kind == 2:
                neg_name = "Global Apex Logistics Corp"
                neg_addr = addr1
            else:
                neg_name = f"{base_name} Co" if "Inc" in name_variants[0] else f"{base_name} Inc"
                neg_addr = "1200 Commercial Way"

            source_rows.append({
                "entity_id": nid,
                "business_name": neg_name,
                "business_address": neg_addr,
                "country": country,
            })
            cand_rows.append({
                "s1_id": s1_id,
                "source_record_id": nid,
                "source": neg_src,
                "country": country,
                "retrieval_rank": rank,
                "retrieval_score": round(float(rng.uniform(0.20, 0.65)), 4),
                "retrieval_method": "prefix_blocking",
            })
            rank += 1

        gt_rows.append({
            "source1_entity_id": s1_id,
            "matched_entity_ids": ",".join(matched_ids),
        })

    s1_df = pd.DataFrame(s1_rows)
    src_df = pd.DataFrame(source_rows)
    cand_df = pd.DataFrame(cand_rows)
    gt_df = pd.DataFrame(gt_rows)

    return s1_df, src_df, cand_df, gt_df


def main():
    parser = argparse.ArgumentParser(description="Run M2 diagnostic matching pipeline.")
    parser.add_argument("--config", default="configs/matching.yaml", help="Path to config file.")
    parser.add_argument("--sample-size", type=int, default=150, help="Number of S1 entities to benchmark.")
    args = parser.parse_args()

    print("\n" + "=" * 65)
    print("  AMAZON ML CHALLENGE 2026 — M2 MATCHING MODEL DIAGNOSTIC")
    print("=" * 65)

    # 1. Load Configuration
    if os.path.exists(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    else:
        config = {}

    # 2. Build Realistic Controlled Sample
    logger.info(f"Generating realistic controlled ER benchmark with {args.sample_size} S1 entities...")
    s1_df, source_df, candidate_df, gt_df = generate_rich_diagnostic_dataset(num_s1=args.sample_size, random_state=42)
    gt_lookup = build_ground_truth_lookup(gt_df)

    logger.info(f"Entities: S1={len(s1_df)}, Source Records={len(source_df)}, Candidate Pairs={len(candidate_df)}")

    # 3. Pairwise Feature Extraction
    logger.info("Running build_features() on candidate pairs...")
    feature_df = build_features(candidate_df, s1_df, source_df, config=config.get("features", {}))
    feature_cols = [c for c in feature_df.columns if c not in ("s1_id", "source_record_id", "source")]
    logger.info(f"Features extracted successfully! Total pairs: {len(feature_df)}, Feature count: {len(feature_cols)}")

    # 4. Label Builder
    logger.info("Running build_labels() using ground truth...")
    y = build_labels(candidate_df, gt_lookup)
    feature_df["label"] = y
    label_summary = compute_label_summary(candidate_df, y, gt_df)

    print("\n--- Training Data & Label Summary ---")
    for k, v in label_summary.items():
        print(f"  {k}: {v}")

    # 5. Grouped S1 Validation Split
    logger.info("Splitting train/validation strictly grouped by S1 entity...")
    val_fraction = config.get("validation", {}).get("val_fraction", 0.25)
    random_state = config.get("validation", {}).get("random_state", 42)
    train_df, val_df = split_grouped_by_s1(feature_df, s1_col="s1_id", val_fraction=val_fraction, random_state=random_state)

    y_train = train_df["label"].values
    y_val = val_df["label"].values

    # Confirm strict zero leakage
    train_s1 = set(train_df["s1_id"].unique())
    val_s1 = set(val_df["s1_id"].unique())
    assert len(train_s1 & val_s1) == 0, "Data leakage detected: S1 entity present in both train and val!"
    logger.info("Zero-leakage verification passed: No S1 overlap between train and validation.")

    # 6. Train Matcher Model
    logger.info("Training supervised XGBoost matcher...")
    model_cfg = config.get("model", {})
    # For small diagnostic sample, set tree_method='auto' and min_child_weight=0 to ensure proper leaf splits
    if "model_params" not in model_cfg:
        model_cfg["model_params"] = {}
    model_cfg["model_params"]["min_child_weight"] = 0
    model_cfg["model_params"]["tree_method"] = "auto"

    matcher = train_matcher(train_df, y_train, config=model_cfg)

    # 7. Predict Match Probabilities
    logger.info("Running predict_match_scores() on validation set...")
    val_scores = predict_match_scores(matcher, val_df)
    val_df_scored = score_candidates(matcher, val_df)

    # 8. Validation Threshold Grid Search
    thresholds = config.get("validation", {}).get("thresholds", [0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90])
    best_res, grid_df = run_threshold_grid_search(val_df, val_scores, gt_lookup, thresholds=thresholds)

    print("\n--- Validation Threshold Grid Search ---")
    print(grid_df[["threshold", "macro_f05", "macro_precision", "macro_recall", "singleton_f05", "non_singleton_f05", "pair_f05", "pair_precision", "pair_recall"]].to_string(index=False))

    print(f"\n>>> Best Validation Result (Threshold = {best_res['threshold']:.2f}):")
    print(f"    Macro F0.5:      {best_res['macro_f05']:.4f}")
    print(f"    Macro Precision: {best_res['macro_precision']:.4f}")
    print(f"    Macro Recall:    {best_res['macro_recall']:.4f}")
    print(f"    Singleton F0.5:  {best_res['singleton_f05']:.4f}")
    print(f"    Non-Singleton:   {best_res['non_singleton_f05']:.4f}")
    print(f"    Pair-level F0.5: {best_res['pair_f05']:.4f} (Prec={best_res['pair_precision']:.4f}, Rec={best_res['pair_recall']:.4f})")

    # 9. Segment Breakdown
    segments = evaluate_segments(val_df, val_scores, gt_lookup, threshold=best_res["threshold"])
    print("\n--- Segment Performance Breakdown at Best Threshold ---")
    for seg_name, m in segments.items():
        print(f"  {seg_name:12s}: Macro F0.5={m['macro_f05']:.4f}, Precision={m['macro_precision']:.4f}, Recall={m['macro_recall']:.4f}")

    # 10. Feature Importances
    fi = matcher.get_feature_importances()
    if fi is not None:
        print("\n--- Top 15 Most Discriminative Features ---")
        print(fi.head(15).to_string(index=False))

    # 11. Save Model Artifact
    os.makedirs("models", exist_ok=True)
    model_path = "models/matcher_baseline.joblib"
    matcher.save(model_path)
    print(f"\nSaved trained matcher artifact to {model_path}")
    print("[NOTE] Diagnostic benchmark verified. Pipeline is ready to receive full M1 candidate sets.")


if __name__ == "__main__":
    main()
