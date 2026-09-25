import json, os, shutil
P=json.load(open("profile.json")); G=json.load(open("gt_stats.json")); X=json.load(open("out/exact_summary.json")); NZ=json.load(open("out/noise.json"))
SIM=json.load(open("out/similarity.json")); T=json.load(open("out/traintest.json")); E=json.load(open("out/extra.json")); K=json.load(open("out/blocking_keys.json"))
B=json.load(open("out/blocking_token.json")); TK=json.load(open("out/topk.json")); DU=json.load(open("out/dups.json")); AP=json.load(open("out/addr_parts.json")); SW=json.load(open("out/suffix_swaps.json"))
names=["train_source1","train_source2","train_source3","train_ground_truth","test_source1","test_source2","test_source3"]
S={}
S["files"]={n:{"bytes":P[n]["file_bytes"],"rows":P[n]["rows"],"cols":len(P[n]["cols"])} for n in names}
def src(n):
    r=P[n]; return {"rows":r["rows"],"unique_ids":r["unique_ids"],"duplicate_id_rows":r["dup_id_rows"],"unique_names":r["unique_names"],"names_appearing_gt1":r["names_appearing_gt1"],"rows_sharing_name":r["rows_with_dup_name"],
        "unique_addresses":r["unique_addresses"],"exact_name_addr_dup_rows":r["rows_in_dup_name_addr"],"missing_name":r["cols_detail"]["business_name"]["blank_or_ws"],"missing_address":r["cols_detail"]["business_address"]["blank_or_ws"],
        "missing_address_pct":r["cols_detail"]["business_address"]["missing_pct"],"country_dist":r["country_dist"],"name_len":r["name_len"],"addr_len":r["addr_len"],"names_non_ascii":r["dq"]["business_name"]["non_ascii_rows"],
        "names_len_le_2":r["dq"]["business_name"]["len_le_2_nonempty"],"addr_control_chars":r["dq"]["business_address"]["has_control"],"addr_mojibake":r["dq"]["business_address"]["mojibake_pat"]}
S["sources"]={n:src(n) for n in names if "ground" not in n}
S["ground_truth"]={k:G[k] for k in ["gt_rows","gt_unique_s1","gt_s1_in_source1","source1_not_in_gt","gt_empty","gt_malformed_match_field","total_pairs","pairs_s2","pairs_s3","dup_matched_ids_globally","dup_within_row","matched_s2_in_source2","matched_s3_in_source3","zero","exactly1","multi","mean_matches","mean_matches_nonzero","median","max","dist_n","dist_n2","dist_n3","both_s2_s3","only_s2","only_s3","s2_matched_frac","s3_matched_frac","s2_unmatched","s3_unmatched"]}
S["ground_truth"]["singleton_rate"]=G["zero"]/G["n_s1"]; S["ground_truth"]["each_s2s3_record_matched_to_at_most_one_s1"]=True; S["ground_truth"]["all_matched_pairs_same_country_pct"]=X["country_same"]
S["exact_match_pct_of_true_pairs"]=X
S["name_noise_taxonomy_pct"]={k:{"pct":v["pct"],"S2":v["pct_S2"],"S3":v["pct_S3"],"US":v["pct_US"],"India":v["pct_India"]} for k,v in NZ["name_taxonomy"].items()}
S["address_noise_taxonomy_pct"]={k:{"pct":v["pct"],"S2":v["pct_S2"],"S3":v["pct_S3"],"US":v["pct_US"],"India":v["pct_India"]} for k,v in NZ["addr_taxonomy"].items()}
S["name_flags_pct"]={"S1":NZ["name_flags_S1side"],"S2_match":NZ["name_flags_matchside_S2"],"S3_match":NZ["name_flags_matchside_S3"]}
S["legal_suffix"]={**NZ["legal_suffix"],"suffix_swap_rates_1M_pairs":SW,"legal_form_conflict_pct":{"true_pairs":0.41,"high_scoring_false_candidates_p>=0.5":5.46}}
S["house_number"]=NZ["house_number"]; S["us_state_city_missing"]={"S2":AP["US_S2"],"S3":AP["US_S3"]}; S["postal_codes_present"]=False
S["similarity"]={"auc_by_negative_type":SIM["auc"],"auc_latin_names_only":SIM["auc_latin_names_only"],"positives_signal_coverage":SIM["positives_signal_coverage"],"diagnostic_hgb":SIM["diagnostic_hgb"]}
S["blocking"]={"single_key":K,"rare_token_df_cutoff":{c:{"recall_by_df_cutoff":B[c]["recall_by_df_cutoff"],"candidates_per_S1":B[c]["candidates_per_S1"],"pool":B[c]["corpus_docs"]} for c in B},"topk_idf_retrieval":TK["recall_at_K"],
   "cross_pairs":{"train_same_country":E["cross_pairs_same_country_train"],"test_same_country":E["cross_pairs_same_country_test"],"train_all":E["cross_pairs_all_train"],"test_all":E["cross_pairs_all_test"]}}
S["singletons"]={"rate":G["zero"]/G["n_s1"],"count":G["zero"],"rate_by_country":E["singleton_rate_by_country"],"s1_attributes_singleton_vs_non":E["singleton_vs_non_S1_attrs"],"danger_false_candidate_share":TK["singleton_danger"],"threshold_sweep_diagnostic":TK["diagnostic_baseline"],"oracle_ceiling_top100":TK["oracle_ceiling_top100"]}
S["decoys"]=T["decoys"]|{"exact_in_source_duplicates_always_matched_same_owner":DU}
S["matched_vs_unmatched_priors"]=T["matched_vs_unmatched_priors"]; S["id_leak_checks"]=T["id_leak"]|{"id_overlap_train_test":T["id_overlap_train_test"]}
S["train_test"]={"country_counts":T["country_counts"],"S1_dup_structure":T["S1_dup_structure"],"S1_legal_token_share":T["S1_legal_token_share"],"oov":{k:v for k,v in T["oov"].items()},"france":{k:{kk:vv for kk,vv in v.items() if kk not in ("sample_names","sample_addresses")} for k,v in T["france"].items()},
   "proxy_S1_has_exact_addr_in_S2S3":T["proxy_S1_has_exact_addr_in_S2S3"],"proxy_share_S2S3_records_with_addr_in_S1":E["proxy_share_of_S2S3_records_with_addr_in_S1"],
   "records_per_S1":{"train_S2":P["train_source2"]["rows"]/P["train_source1"]["rows"],"train_S3":P["train_source3"]["rows"]/P["train_source1"]["rows"],"test_S2":P["test_source2"]["rows"]/P["test_source1"]["rows"],"test_S3":P["test_source3"]["rows"]/P["test_source1"]["rows"]},
   "inferred_test_unmatched_share_S2S3":0.41,"train_unmatched_share":{"S2":1-G["s2_matched_frac"],"S3":1-G["s3_matched_frac"]},"test_S1_name_seen_in_train_pct":T["S1_test_norm_name_seen_in_train_S1_pct_by_country"]}
S["performance"]={"retrieval_ms_per_S1_single_core":{"US":45,"India":78},"pair_feature_us_per_pair":58,"est_test_retrieval_core_hours":30,"pandas_ram_gb_all_seven_files":7.9,"candidate_pairs_test_by_K":{str(k):1732544*k for k in (20,30,50,100)}}
S["notes"]={"diagnostic_baseline":"macro-F0.5 ~0.89 (US 0.94, India 0.85) at p>=0.6 with top-100 IDF retrieval + 19 features + HistGradientBoosting trained on 1,200 entities/country; diagnostic only, not a final model","samples":"noise taxonomies: 500k pairs; similarity: 150k positives; blocking: 1,500 S1/country candidates, 120k-200k positive pairs; top-K: 2,400 S1/country"}
json.dump(S,open("/Users/sanjana1106/Desktop/MLChallenge/dataset_analysis_summary.json","w"),indent=1,ensure_ascii=False,default=str)
os.makedirs("/Users/sanjana1106/Desktop/MLChallenge/analysis_scripts",exist_ok=True)
import glob
for f in glob.glob("s*.py")+["util.py"]+glob.glob("make_*.py"): shutil.copy(f,"/Users/sanjana1106/Desktop/MLChallenge/analysis_scripts/")
parts=[open(f"report_{p}.md").read() for p in ("A","B1","B2","C","D")]
open("/Users/sanjana1106/Desktop/MLChallenge/dataset_analysis_report.md","w").write("\n\n".join(parts)+"\n")
print("ok",os.path.getsize("/Users/sanjana1106/Desktop/MLChallenge/dataset_analysis_summary.json")//1024,"KB json;",os.path.getsize("/Users/sanjana1106/Desktop/MLChallenge/dataset_analysis_report.md")//1024,"KB md")
