# Part A: builds sections 0-5 of the report; each part appends to a list saved as pickle-free text file.
import json, pandas as pd, numpy as np, os
P=json.load(open("profile.json")); G=json.load(open("gt_stats.json")); X=json.load(open("out/exact_summary.json"))
E=json.load(open("out/extra.json")); T=json.load(open("out/traintest.json")); D=json.load(open("out/dups.json")); AP=json.load(open("out/addr_parts.json"))
out=[]
def w(s=""): out.append(s)
def esc(x,n=70):
    x=str(x).replace("|","\\|").replace("\n"," ").replace("\t"," ")
    x=x.replace("\x1a","<0x1A>")
    return (x[:n]+"…") if len(x)>n else x
def tbl(h,rows):
    w("| "+" | ".join(h)+" |"); w("|"+"|".join(["---"]*len(h))+"|")
    for r in rows: w("| "+" | ".join(esc(c,90) if isinstance(c,str) else (f"{c:,}" if isinstance(c,int) else str(c)) for c in r)+" |")
    w()
def mib(b): return f"{b/2**20:,.1f} MiB"
names=["train_source1","train_source2","train_source3","train_ground_truth","test_source1","test_source2","test_source3"]
w("# Amazon ML Challenge 2026 — Business Entity Resolution: Dataset Reconnaissance Report")
w()
w("**Scope:** reconnaissance only. No final model was trained, the original dataset was not modified (parquet/JSON caches and scripts live outside `dataset/`), and no external data, APIs, geocoders or web lookups were used. Everything below is measured from the supplied files.")
w()
w("**Environment:** macOS, 10 cores, 16 GB RAM, Python 3.13, pandas 2.2, scikit-learn 1.6, rapidfuzz 3.14, scipy. Files were read with `sep=\"\\t\"`, `quoting=csv.QUOTE_NONE`, `dtype=str`, `keep_default_na=False` (see §13 for why the last two matter).")
w()
w("**How to read the numbers.** Full-file statistics (sizes, schema, duplicates, missingness, ground-truth structure, exact-match rates over all 7.64 M positive pairs) are exact. Statistics marked *(sample)* come from random samples of pairs/entities (size stated) — the sampling error is small for the headline rates but not for rare categories. Blocking candidate counts come from 1,500-entity samples per country; the ‘diagnostic’ baseline in §11/§12 uses 1,200 training and 1,200 evaluation entities per country and is a **measurement instrument, not a proposed final model**.")
w()
w("**Normalizations used throughout** (all deterministic, no external resources):")
w()
w("- `basic`: lowercase → every ASCII punctuation/symbol char (and common Unicode quotes/dashes/danda `।`) replaced by a space → whitespace collapsed → strip. Accents and native scripts are kept.")
w("- `ext` (extended): `basic` + `&`→`and` + Latin-diacritic folding (NFKD, drop U+0300–036F only, so Indic vowel signs are untouched) + abbreviation expansion (`pvt→private, ltd→limited, corp→corporation, inc→incorporated, co→company`; for addresses also `rd→road, st→street, ave/av→avenue, dr→drive, ln→lane, ct→court, cir→circle, blvd/bd→boulevard, hwy→highway, ter→terrace, pl→place, apt→apartment, fl→floor, nr→near, opp→opposite, r→rue, n/s/e/w→compass` …) + for addresses, `NULL`/`N/A` placeholders dropped. `sorted` = `ext` with tokens sorted (removes word/component order).")
w()
w("Files produced next to this report: `dataset_analysis_report.md`, `dataset_analysis_summary.json`, and `analysis_scripts/` (the main pipeline scripts `s00`–`s11` plus the report builders; a few one-off checks — suffix-swap rates, the legal-form-conflict rate, the Unicode/control-character scan, the singleton-decoy inspection — were run interactively and are **not** saved as scripts).")
w()
# ---------------- 1
w("## 1. FILE SIZES"); w()
rows=[]; tb=0
for n in names:
    r=P[n]; sp=n.split("_")[0]; rows.append([f"{sp}/{n}.tsv",f"{r['file_bytes']:,} B ({mib(r['file_bytes'])})",r["rows"],len(r["cols"])]); tb+=r["file_bytes"]
tbl(["file","size","data rows (excl. header)","columns"],rows)
w(f"Total on disk: {tb/2**30:.2f} GiB (train {sum(P[n]['file_bytes'] for n in names[:4])/2**30:.2f} GiB, test {sum(P[n]['file_bytes'] for n in names[4:])/2**30:.2f} GiB). Raw-level integrity check (awk): every line of every file has exactly the expected number of tab-separated fields (4, or 2 for the ground truth), no CR characters, no BOM issues; the files decode as strict UTF-8 with zero errors. A handful of lines contain a literal `\"` (train: 4 in S1, 6 in S2; test: 134/349/330) — pandas' default quote handling would mis-parse those, hence `QUOTE_NONE`.")
w()
w(f"Test-set scale: **{P['test_source1']['rows']:,} Source-1 entities** must each get a row; the candidate pool is {P['test_source2']['rows']:,} S2 + {P['test_source3']['rows']:,} S3 = {P['test_source2']['rows']+P['test_source3']['rows']:,} records.")
w()
# ---------------- 2
w("## 2. SCHEMA"); w()
w("All columns are read as strings; ‘inferred type’ is what the values actually look like. ‘Missing’ = empty / whitespace-only string. **No file contains NaN-style missingness in `entity_id`, `business_name` or `country`; the only truly empty field is `business_address` (S2/S3) and `matched_entity_ids` (singletons in ground truth).** Placeholder strings inside addresses (`NULL`, `<NULL>`, `N/A`) are *not* counted as missing here — see §7/§13.")
w()
types={"entity_id":"string key `S<k>-<integer 1–9 digits>`","business_name":"free text (Latin + Indic scripts + accents)","business_address":"free text, comma-separated components (Latin + Indic scripts)","country":"categorical string","source1_entity_id":"string key `S1-<int>`","matched_entity_ids":"comma-separated list of `S2-`/`S3-` keys"}
for n in names:
    r=P[n]; w(f"### {n}.tsv — {r['rows']:,} rows × {len(r['cols'])} cols"); w()
    rows=[]
    for c,d in r["cols_detail"].items():
        ex=" ; ".join(esc(e,45) for e in d["examples"][:3]); rows.append([c,types.get(c,""),ex,d["blank_or_ws"],f"{d['missing_pct']:.3f}%",d["unique"]])
    tbl(["column","inferred type","example values","missing","missing %","unique"],rows)
# ---------------- 3/4
def col(n,k,sub=None):
    v=P[n][k]; return v if sub is None else v[sub]
w("## 3. SOURCE 1 ANALYSIS (train_source1.tsv)"); w()
s=P["train_source1"]
w(f"- **Total entities:** {s['rows']:,}; **unique entity IDs:** {s['unique_ids']:,}; **duplicate IDs:** {s['dup_id_rows']} (all IDs match `^S1-\\d+$`; numeric part 217 … 999,998,822, i.e. random 1–9-digit integers).")
w(f"- **Unique business names:** {s['unique_names']:,} ({100*s['unique_names']/s['rows']:.1f}% of rows); {s['unique_names_lower']:,} case-insensitively. **{s['names_appearing_gt1']:,} distinct names occur more than once, covering {s['rows_with_dup_name']:,} rows ({100*s['rows_with_dup_name']/s['rows']:.1f}% of S1).** ‘Deduplicated’ therefore means *no exact duplicate records*, not *unique names*: 0 rows share the same (name, address); but the same name is reused by different entities at different addresses (max 253 entities named `Primary Care Group`). Top repeated names: "+", ".join(f"`{k}` ×{v}" for k,v in list(s['top_names'].items())[:6])+".")
w(f"- **Unique addresses:** {s['unique_addresses']:,}; {s['addresses_appearing_gt1']:,} addresses are shared by ≥2 entities ({s['rows_with_dup_address']:,} rows, {100*s['rows_with_dup_address']/s['rows']:.1f}%; max 14 entities at one address).")
w(f"- **Missing name / address / country:** 0 / 0 / 0.")
w(f"- **Country distribution:** "+", ".join(f"{k} {v:,} ({100*v/s['rows']:.1f}%)" for k,v in s['country_dist'].items())+".")
nl,al=s["name_len"],s["addr_len"]
tbl(["length (chars)","min","mean","std","p5","p25","median","p75","p95","p99","max"],[["name",nl["min"],nl["mean"],nl["std"],nl["p5"],nl["p25"],nl["p50"],nl["p75"],nl["p95"],nl["p99"],nl["max"]],["address",al["min"],al["mean"],al["std"],al["p5"],al["p25"],al["p50"],al["p75"],al["p95"],al["p99"],al["max"]]])
w(f"Mean tokens: name {s['name_tokens']}, address {s['addr_tokens']}. Country matters for address shape: US addresses average 35 chars / 5.9 tokens (`2620 Sawmill Road, Fort Recovery, OH`), India 78 chars / 11.2 tokens (multi-component, landmarks, sub-localities).")
w()
w("## 4. SOURCE 2 AND SOURCE 3 (same analysis; S1 repeated for comparison)"); w()
def r3(f): 
    return [P[f"train_source{k}"][f] if f in P[f"train_source{k}"] else "" for k in (1,2,3)]
rows=[]
def add(label,fn): rows.append([label]+[fn(P[f"train_source{k}"]) for k in (1,2,3)])
add("total rows",lambda r:r["rows"]); add("unique IDs",lambda r:r["unique_ids"]); add("duplicate IDs",lambda r:r["dup_id_rows"])
add("unique names",lambda r:r["unique_names"]); add("unique names (case-insens.)",lambda r:r["unique_names_lower"])
add("names occurring >1×",lambda r:r["names_appearing_gt1"]); add("rows sharing a name",lambda r:r["rows_with_dup_name"])
add("unique addresses (incl. '')",lambda r:r["unique_addresses"]); add("rows sharing a non-trivial address",lambda r:r["rows_with_dup_address"]-(r["cols_detail"]["business_address"]["empty"] if r["cols_detail"]["business_address"]["empty"]>0 else 0))
add("exact (name,addr) duplicate rows",lambda r:r["rows_in_dup_name_addr"])
add("missing name",lambda r:r["cols_detail"]["business_name"]["blank_or_ws"]); add("missing address",lambda r:r["cols_detail"]["business_address"]["blank_or_ws"]); add("missing address %",lambda r:f"{r['cols_detail']['business_address']['missing_pct']:.2f}%"); add("missing country",lambda r:r["cols_detail"]["country"]["blank_or_ws"])
add("country US",lambda r:f"{r['country_dist']['US']:,} ({100*r['country_dist']['US']/r['rows']:.1f}%)"); add("country India",lambda r:f"{r['country_dist']['India']:,} ({100*r['country_dist']['India']/r['rows']:.1f}%)")
for k in ("min","mean","p50","p95","p99","max"): add(f"name length {k}",lambda r,k=k:r["name_len"][k])
for k in ("min","mean","p50","p95","p99","max"): add(f"address length {k} (non-empty)",lambda r,k=k:r["addr_len_nonempty"][k])
add("mean name tokens",lambda r:r["name_tokens"]); add("mean address tokens (incl. empties)",lambda r:r["addr_tokens"])
add("names with non-ASCII chars",lambda r:r["dq"]["business_name"]["non_ascii_rows"]); add("addresses with non-ASCII chars",lambda r:r["dq"]["business_address"]["non_ascii_rows"])
add("names with double space",lambda r:r["dq"]["business_name"]["double_space"])
tbl(["metric","S1","S2","S3"],rows)
w("**Reading it:** S2 and S3 are ~2.3–2.4× the size of S1 and are far *less* clean: 3.3–3.4% empty addresses, 11%+ of names contain a double space, ~15% of S2 names and ~11.5% of S3 names contain non-ASCII characters (native Indic script for India records, injected accents such as `Ínnovative` for US records; see §6), and none of S1's names do, and 50 k (S2) / 37 k (S3) rows are exact (name, address) duplicates of another row *in the same source* (§5/§13 — these are always true matches). Top repeated S2/S3 names are generic (`Primary Care` ×320/421, `CC` ×302/387) — they are noise-degraded variants of many different S1 entities, so name alone is not identifying.")
w()
w("Top country notes — S2 by country: "+", ".join(f"{k} {v:,}" for k,v in P['train_source2']['country_dist'].items())+"; S3: "+", ".join(f"{k} {v:,}" for k,v in P['train_source3']['country_dist'].items())+". Missing address by country — S2: "+", ".join(f"{k} {v:,}" for k,v in P['train_source2']['missing_addr_by_country'].items())+"; S3: "+", ".join(f"{k} {v:,}" for k,v in P['train_source3']['missing_addr_by_country'].items())+".")
w()
# ---------------- 5
w("## 5. GROUND TRUTH ANALYSIS (train_ground_truth.tsv)"); w()
w(f"- Rows: {G['gt_rows']:,}, one per S1 entity; **{G['gt_unique_s1']:,} unique `source1_entity_id`s; all {G['gt_s1_in_source1']:,} exist in train_source1 and every train S1 entity appears in the ground truth (0 missing, 0 extra)**. Row order differs from the S1 file order.")
w(f"- **Malformed rows: 0.** All non-empty match fields match `S[23]-\\d+(,S[23]-\\d+)*` (no spaces after commas, no stray tokens); {G['gt_empty']:,} rows have an empty match list (singletons). Within each list all S2 IDs come before S3 IDs.")
w(f"- **Every matched ID exists** in the right source: {G['matched_s2_in_source2']:,}/{G['pairs_s2']:,} S2 IDs are in train_source2 and {G['matched_s3_in_source3']:,}/{G['pairs_s3']:,} S3 IDs in train_source3.")
w(f"- **Duplicate matched IDs: none** — not within a row (0), and not across rows (0 IDs shared between two S1 entities; {G['unique_matched_ids']:,} unique matched IDs = {G['total_pairs']:,} pairs). **So each S2/S3 record belongs to at most one S1 entity (the relation is a partition, with ~26% of S2/S3 records left unmatched).** This is a strong structural constraint (§15).")
w()
tbl(["metric","value"],[["Source-1 entities",G["n_s1"]],["with zero matches (singletons)",f"{G['zero']:,} ({100*G['zero']/G['n_s1']:.3f}%)"],["exactly 1 match",f"{G['exactly1']:,} ({100*G['exactly1']/G['n_s1']:.2f}%)"],["multiple matches (≥2)",f"{G['multi']:,} ({100*G['multi']/G['n_s1']:.2f}%)"],
 ["mean matches / S1 entity",f"{G['mean_matches']:.3f} (excluding singletons: {G['mean_matches_nonzero']:.3f})"],["median matches",G["median"]],["max matches",G["max"]],
 ["total positive pairs",G["total_pairs"]],["…from S2",f"{G['pairs_s2']:,} (mean {G['mean_n2']:.2f}/entity, max {G['max_n2']})"],["…from S3",f"{G['pairs_s3']:,} (mean {G['mean_n3']:.2f}/entity, max {G['max_n3']})"],
 ["entities with matches in BOTH S2 and S3",f"{G['both_s2_s3']:,} ({100*G['both_s2_s3']/G['n_s1']:.1f}%; {100*G['both_s2_s3']/(G['n_s1']-G['zero']):.1f}% of non-singletons)"],["only S2",f"{G['only_s2']:,}"],["only S3",f"{G['only_s3']:,}"],
 ["S2 records matched / unmatched",f"{G['matched_s2_unique']:,} ({100*G['s2_matched_frac']:.1f}%) / {G['s2_unmatched']:,}"],["S3 records matched / unmatched",f"{G['matched_s3_unique']:,} ({100*G['s3_matched_frac']:.1f}%) / {G['s3_unmatched']:,}"]])
dn=G["dist_n"]; tot=G["n_s1"]
w("**Distribution of total matches per S1 entity** (with S2/S3 splits):"); w()
d2,d3=G["dist_n2"],G["dist_n3"]
tbl(["# matches","S1 entities","%","","# S2 matches","S1 entities","# S3 matches","S1 entities"],[[k,v,f"{100*v/tot:.2f}%","",(list(d2)[i] if i<len(d2) else ""),(list(d2.values())[i] if i<len(d2) else ""),(list(d3)[i] if i<len(d3) else ""),(list(d3.values())[i] if i<len(d3) else "")] for i,(k,v) in enumerate(dn.items())])
w("The distribution is unimodal around 3–4 matches (P(0)=5.6%, P(1)=5.4%, mode 3), roughly ‘each S1 entity is expanded into 0–5 S2 and 0–6 S3 noisy variants’. Only 5.4% of entities have a single match; predicting a small number of matches is *not* the right prior — most entities have 2–5. Matches per entity, by country (train): see §10.")
w()
open("report_A.md","w").write("\n".join(out))
print(len(out))
