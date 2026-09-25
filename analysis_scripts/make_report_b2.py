import json, pandas as pd, numpy as np
SIM=json.load(open("out/similarity.json")); T=json.load(open("out/traintest.json")); G=json.load(open("gt_stats.json"))
out=[]
def w(s=""): out.append(s)
def esc(x,n=110):
    x=str(x).replace("|","\\|").replace("\n"," ").replace("\t"," ").replace("\x1a","<0x1A>"); return (x[:n]+"…") if len(x)>n else x
def tbl(h,rows):
    w("| "+" | ".join(h)+" |"); w("|"+"|".join(["---"]*len(h))+"|")
    for r in rows: w("| "+" | ".join(esc(c) if isinstance(c,str) else (f"{c:,}" if isinstance(c,int) else str(c)) for c in r)+" |")
    w()
# ---------------- 9
w("## 9. NAME (AND ADDRESS) SIMILARITY ANALYSIS — positives vs negatives"); w()
c=SIM["counts"]
w("**Design** *(samples; features are computed on `ext`-normalised strings)*. Positives: 150,000 random true pairs. Negatives (four kinds, because the choice of negative decides how ‘separable’ things look):")
w()
tbl(["negative type","n","how it is built"],[
 ["`random_same_country`",c["random_same_country"],"random S1 entity × random S2/S3 record of the same country (includes unmatched records). *This is the easy case.*"],
 ["`hard_same_block`",c["hard_same_block(country+name[:4])"],"S1 × a non-matching S2/S3 record that shares country + first 4 chars of the first name token (a typical name-blocking false candidate)"],
 ["`hard_same_name_diff_entity`",c["hard_same_name_diff_entity"],"S1 entity *a* × a true match of a **different** S1 entity *b* that has the same normalised name as *a* (name-twins, 38% of S1)"],
 ["`hard_same_addr_diff_entity`",c["hard_same_addr_diff_entity"],"same as above but *a* and *b* share the same normalised address"]])
w("No sampled negative is a true pair (checked: 0). Features: character similarity (`fuzz.ratio`, `partial_ratio`), token-order-insensitive (`token_sort_ratio`, `token_set_ratio`), edit-distance (normalised Levenshtein), Jaro-Winkler, token Jaccard / overlap-coefficient, char-3-gram Jaccard, and TF-IDF cosine (word-level and char-3-gram; vectorisers fitted on a 600k-string sample of S1+S2+S3). Address features are the same plus a house-number-equality flag, and are `NaN` when the matched address is empty.")
w()
M,Md=SIM["mean_by_kind"],SIM["median_by_kind"]
feats=["n_ratio","n_tsort","n_tset","n_jw","n_lev","n_jacc","n_c3jacc","n_tfidf_word","n_tfidf_char","a_tsort","a_tset","a_jacc","a_tfidf_word","a_tfidf_char","a_housenum_eq"]
kinds=["positive","random_same_country","hard_same_block(country+name[:4])","hard_same_name_diff_entity","hard_same_addr_diff_entity"]
tbl(["feature (mean)","positive","random neg","hard: same block","hard: same name","hard: same addr"],[[f]+[f"{M[k][f]:.3f}" for k in kinds] for f in feats])
w("Positives: name `token_sort_ratio` mean 0.80 / median 0.89 (p25 0.73, p5 0.11) vs 0.34 for random negatives; address `token_set_ratio` mean 0.93 vs 0.37. The **positive distribution is heavily skewed with a fat lower tail**: ~66% of positives score ≥0.8 on name, but ~5% are near zero (p5 = 0.11) and ~10% are below 0.43 (native script / unrelated alias / heavy edits) — the p5 value is essentially an ‘unrelated’ name.")
w()
A=SIM["auc"]
neg=["random_same_country","hard_same_block(country+name[:4])","hard_same_name_diff_entity","hard_same_addr_diff_entity","ALL_neg"]
w("**Single-feature ROC-AUC, positive vs each negative type** (0.5 = no separation; the ‘same addr’ column is meaningless for address features by construction):"); w()
tbl(["feature","vs random","vs same-block","vs same-name/diff-entity","vs same-addr/diff-entity","vs all negatives"],[[f]+[f"{A[n][f]:.3f}" for n in neg] for f in feats])
A2=SIM["auc_latin_names_only"]
w("The same restricted to pairs whose matched name is **not** native script (Latin-script names only):"); w()
tbl(["feature","vs random","vs same-block","vs same-name/diff-entity","vs same-addr/diff-entity"],[[f]+[f"{A2[n][f]:.3f}" for n in neg[:4]] for f in ["n_tsort","n_tset","n_jw","n_lev","n_jacc","n_c3jacc","n_tfidf_word","n_tfidf_char"]])
w("**What this says**")
w()
w("1. **Names separate well against random and blocked negatives** (AUC 0.90–0.94 over all positives; **0.95–0.99 when the matched name is Latin-script**, with char-3-gram / TF-IDF-char / partial-ratio best and token-Jaccard / Levenshtein worst), but **only ~0.83 against all negatives overall** because native-script and alias names (~11% of positives) score like negatives. Character-n-gram measures beat token measures (0.987 vs 0.948 for TF-IDF char vs word against random negatives, Latin names) — the typos/leet/accents break exact tokens but not most trigrams.")
w("2. **Names cannot separate same-name twins** (AUC 0.50–0.56 for every name feature): when two S1 entities share a name, only the address can tell whose record it is. 38% of S1 rows are in such groups.")
w("3. **Addresses separate at AUC 0.94–0.96 against random/blocked negatives** (TF-IDF, token-set and Jaccard best; Levenshtein weakest 0.90 because of reordering) and are the *only* signal against name-twins (0.945–0.985) — but fail on the ~4.4% empty addresses and are blind to same-address twins (0.50), where the name decides. **The two fields are complementary, not redundant.**")
w("4. Coverage of positives by simple thresholds: name `token_sort_ratio`≥0.8 for 66.4%; address ≥0.8 for 74.2%; **either** for 91.3%; neither for 8.75%; name<0.5 but address ≥0.7 for 9.0% (rescued only by address); name≥0.8 but address <0.5/empty for 5.3% (rescued only by name); **name<0.5 and address weak/empty for only ~1.0%** (near-hopeless pairs).")
w()
w("**Best single-feature F0.5 at an optimal threshold** (balanced positives/negatives sample, so absolute precision is *optimistic* vs real prevalence): name `token_set_ratio` reaches F0.5 = 0.96 vs random negatives but only 0.73 vs same-name twins; address TF-IDF reaches 0.987 vs random and 0.982 vs same-block, but 0.714 vs same-address twins.")
w()
D=SIM["diagnostic_hgb"]
w("**Combined separability (diagnostic only — a small gradient-boosting model on the ~20 features above, trained on 70% of these pairs, evaluated on 30%; this is a probe of separability, not a proposed model):**"); w()
tbl(["evaluated against","AUC","precision @p≥0.5 / 0.8 / 0.9 / 0.95","recall @ same thresholds"],[[k,D[k]["auc"],
  " / ".join(f"{D[k][f'thr{t}']['precision']:.3f}" for t in (0.5,0.8,0.9,0.95)),
  " / ".join(f"{D[k][f'thr{t}']['recall']:.3f}" for t in (0.5,0.8,0.9,0.95))] for k in ["random_same_country","hard_same_block(country+name[:4])","hard_same_name_diff_entity","hard_same_addr_diff_entity"]])
r=D["positive_recall@0.9_by"]
w(f"Combined AUC 0.998 over all negatives. **Recall at p≥0.9 on positives: S2 {r['src']['2']}, S3 {r['src']['3']}; US {r['country']['US']}, India {r['country']['India']}; native-script names {r['native_script_name']['1.0']} vs Latin {r['native_script_name']['0.0']}; empty matched address {r['addr_empty']['1.0']} (!) vs non-empty {r['addr_empty']['0.0']}.** So the pairs are *highly separable when both fields are present* (>95% recall at very high precision); the unrecoverable mass is (i) empty-address positives (4.4%; recall ~6%, name only) and (ii) native-script/alias names. Caveat: these negatives are sampled, not the real candidate stream; §11–12 repeat the exercise on realistic retrieved candidates, where separation is noticeably harder (24% of singletons still receive a false candidate with p≥0.5).")
w()
# ---------------- 10
w("## 10. COUNTRY ANALYSIS"); w()
cc=T["country_counts"]
rows=[]
for sp in ("train","test"):
    for s in (1,2,3):
        d=cc[f"{sp}_S{s}"]; tot=sum(d.values())
        rows.append([f"{sp} S{s}",tot]+[f"{d.get(k,0):,} ({100*d.get(k,0)/tot:.1f}%)" for k in ("US","India","France")])
tbl(["file","rows","US","India","France"],rows)
w("- **Countries in training data: exactly two, `US` and `India`.** The `country` field is fully populated (0 missing) with clean, canonical labels in every file — no case/spacing variants (`us`, `USA`, `IN`, …); `repr()` of the unique values shows only `'US'`, `'India'` (train) and additionally `'France'` (test S1/S2/S3). **France exists only in test** (259,452 S1 / 703,378 S2 / 731,615 S3 = 15.0% / 14.4% / 14.4% of test records) and has no labelled example. Treat `country` as an open string key.")
w("- **Country distribution by source is stable across S1/S2/S3** in train (US ≈ 60.0%, India ≈ 40.0%): S1 59.98% US, S2 59.92%, S3 59.98%; the test US/India split is 38.3/46.8 with France 15.0% for S1, 38.3/47.3/14.4 for S2 and 38.3/47.3/14.4 for S3 — i.e. **test is not a shrunken train**: the US share falls from 60% to 38%, India rises 40%→47%, and France adds 14–15%.")
import numpy as np
F=pd.read_parquet("cache/pair_exact_flags.parquet",columns=["c1","src","country_same"])
pc=F.groupby(["c1","src"]).size().unstack()
gc=pd.read_parquet("cache/gt_counts.parquet").set_index("sid").n; s1=pd.read_parquet("cache/train_source1.parquet",columns=["entity_id","country"])
s1["n"]=gc.reindex(s1.entity_id.values).values
bc=s1.groupby("country").n.agg(["size","mean",lambda x:(x==0).mean(),"max"])
tbl(["S1 country","S1 entities","positive pairs to S2","to S3","mean matches / entity","singleton rate","max"],[[k,int(bc.loc[k,"size"]),int(pc.loc[k,2]),int(pc.loc[k,3]),f"{bc.loc[k,'mean']:.3f}",f"{100*bc.loc[k,'<lambda_0>']:.2f}%",int(bc.loc[k,'max'])] for k in bc.index])
w(f"- **Matched pairs by country:** every one of the {len(F):,} training positives has `country(S1) == country(S2/S3)` (**{100*F.country_same.mean():.2f}%**; 0 pairs cross countries) — there are **no examples where the country differs**. Same-country blocking is therefore lossless in train and cuts the candidate pool by ~40–60% (US pool 6.19 M, India 4.13 M in train; test pools US 3.82 M, India 4.72 M, France 1.43 M).")
w("- Match density is the same in both training countries: 3.46 matches per entity, 5.6% singletons, and the fraction of S2/S3 records that are matched is 73.4%/74.6% in both US and India (S2 US 73.36%, India 73.37%; S3 US 74.62%, India 74.65%). The generator is country-symmetric; **only noise style differs** (India: native-script names/addresses, longer addresses with landmarks; US: ZIP-less street addresses, `PO Box`/`PMB`).")
w("- **France (test only) — what can be learned without labels** (§14 has more): Latin script with heavy accent use (15.7% of S1 names and 73k S1 addresses contain accented letters), French legal forms (`SARL` 36%, `SAS` 25%, `EURL`, `SASU`, `SCI`, `SA`, `SNC`; `S.A.S`/`S.A.R.L` dotted variants only in S2/S3), association-type words (`Club`, `Amicale`, `Comité`, `École`, `Maison`, `Établissements`/`Ets`) and short template names (mean 19.4 chars vs 24 in train). Addresses use `Rue/Avenue/Allée/Boulevard/Impasse` with `R.`, `Av`, `Bd`, `Pl.` abbreviations in S2/S3 and roughly 15 cities account for most addresses (Bordeaux, Nantes, Lille, Tourcoing, Dunkerque, Roubaix, Calais, Saint-Nazaire, Pessac, La Teste-de-Buch, Mérignac, Lège-Cap-Ferret, Pornic, Saint-Herblain…) with region names (Hauts-de-France, Nouvelle-Aquitaine, Pays de la Loire) — **so city/region tokens carry almost no discriminative information in France; street name + number must do the work.**")
w()
w("**Consequence for design:** never one-hot or filter on `{US, India}`; block within `country` as an opaque string; and make every learned component (abbreviation dictionaries, IDF weights, thresholds) either country-conditional with a *global fallback* or unsupervised-fit on test text, because a model trained only on US/India will have seen none of France’s vocabulary (53% of French S1 name tokens and 37% of address tokens never occur in any training file).")
w()
open("report_B2.md","w").write("\n".join(out)); print(len(out))
