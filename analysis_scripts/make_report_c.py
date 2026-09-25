import json, pandas as pd, numpy as np
P=json.load(open("profile.json")); T=json.load(open("out/traintest.json")); E=json.load(open("out/extra.json")); K=json.load(open("out/blocking_keys.json")); B=json.load(open("out/blocking_token.json"))
TK=json.load(open("out/topk.json")); DQ=json.load(open("out/dq_extra.json")); DU=json.load(open("out/dups.json"))
out=[]
def w(s=""): out.append(s)
def esc(x,n=120):
    x=str(x).replace("|","\\|").replace("\n"," ").replace("\t"," ").replace("\x1a","<0x1A>"); return (x[:n]+"…") if len(x)>n else x
def tbl(h,rows):
    w("| "+" | ".join(h)+" |"); w("|"+"|".join(["---"]*len(h))+"|")
    for r in rows: w("| "+" | ".join(esc(c) if isinstance(c,str) else (f"{c:,}" if isinstance(c,int) else str(c)) for c in r)+" |")
    w()
def f(x,d=1): return f"{x:,.{d}f}"
# ---------------- 11
w("## 11. CANDIDATE / BLOCKING INSIGHTS"); w()
w("### 11.1 Size of the problem"); w()
tr,te=E["cross_pairs_same_country_train"],E["cross_pairs_same_country_test"]
tbl(["","all pairs S1×(S2∪S3)","same-country pairs","per S1 entity (same-country pool)"],[
 ["train",f"{E['cross_pairs_all_train']:.3e}",f"{tr['total']:.3e}",f"US {6186873:,} · India {4133346:,} records"],
 ["test",f"{E['cross_pairs_all_test']:.3e}",f"{te['total']:.3e}",f"India {2312565+2405000:,} · US {1871330+1945701:,} · France {703378+731615:,} records"]])
w("Brute force is impossible (10¹³ pairs). The numbers below evaluate each blocking family on **training positives (recall) and on the real S2∪S3 pool of the same country (candidate counts)**. Test pools are the same order of magnitude (US 38% smaller, India 14% larger than train) so candidate counts transfer approximately. Recall = share of the true (S1, S2/S3) pairs that would survive the blocker; ‘cands’ = number of S2+S3 records returned per S1 entity.")
w()
w("### 11.2 Same-country blocking"); w()
w("Recall **100.00%** (7,638,365/7,638,365 training positives are same-country); pool per S1 drops from 10.3 M to 6.2 M (US) / 4.1 M (India) in train. Necessary and free, but leaves 10⁶ candidates per entity — it is a *filter*, not a blocker. Because France has no training data, the safe implementation is `country` string equality (open set).")
w()
w("### 11.3 Single-key blocking (exact key equality; recall from 200k sampled positives per country, candidates exact over the full pool)"); w()
sel=["name_first_token","name_first3chars(nospace)","name_first4chars(nospace)","name_first6chars(nospace)","name_first_token[:4]","name_min_token_alpha","name_sorted_first2tokens","addr_first_number","state","state+name_first3","state+addr_first_number","addr_first_number+name_first3","addr_first_number+name_first_token[:4]"]
for c in ("US","India"):
    rows=[[k,f"{100*K[c][k]['recall']:.1f}%",f(K[c][k]['cands_mean'],0),f(K[c][k]['cands_median'],0),f(K[c][k]['cands_p99'],0),f"{K[c][k]['s1_with_zero_cands_pct']}%"] for k in sel if k in K[c]]
    w(f"**{c}** (S1 entities {K[c]['N1']:,}; pool {K[c]['N23']:,})"); w()
    tbl(["key (within country)","recall","mean cands / S1","median","p99","S1 with 0 cands"],rows)
w("**Take-aways.** (1) *First characters/tokens of the normalised name* are cheap and reach 81–87% recall in the US but return **≈5,500–14,000 candidates per S1** (the name vocabulary is small and templated: `Primary`, `Global`, …); in India recall is only 59–64% because ~18% of India positives have native-script names whose first characters cannot match a Latin S1 name. Longer prefixes trade recall for size only slowly (US first-6-chars: 82% recall, still 5,452 mean/476 median). (2) `min-token` / `longest-token` keys are hopeless (skew: up to 843k candidates). (3) **House number alone**: 77% (US) / 66% (India) recall — bounded by number mutation (~17%) and the 4.4% empty addresses — with 2,750 (US) / 39,000 (India) candidates. (4) The only keys that give tiny candidate lists are **conjunctions** (number + name prefix: 6–8 candidates in the US) and they lose a third to 60% of the positives (recall 66% US, 41% India). (5) `state` reaches 93% recall in the US (it loses the 4.7% empty addresses and the ~15% of S3 addresses that drop the state) but returns 272k candidates; state combined with a name prefix is 80% / 611 candidates. **No single key works; the signal is spread across name tokens and address tokens, with either alone failing on 25–35% of positives.**")
w()
w("### 11.4 Rare-token blocking (inverted index; ‘shares ≥1 token whose document-frequency ≤ T within the same-country S2∪S3 pool’)"); w()
w("Document frequencies are measured on the full same-country pool (US 6.19 M docs, India 4.13 M docs). Recall is on 120k positives per country; candidate counts are exact for 1,500 sampled S1 entities per country. ‘name_p4’ = 4-character token prefixes of the name (typo-tolerant); ‘addr’ = address tokens; ‘∪’ = union of the two candidate sets; ‘∧’ = must share both a name token and an address token.")
w()
for c in ("US","India"):
    R=B[c]; rows=[]
    for Tt in ["20","100","300","1000","3000","10000"]:
        rc=R["recall_by_df_cutoff"][Tt]; cd=R["candidates_per_S1"].get(Tt)
        rows.append([Tt,f"{100*rc['name']:.1f}%",f"{100*rc['name_p4']:.1f}%",f"{100*rc['addr']:.1f}%",f"**{100*rc['name_p4|addr']:.1f}%**",(f"{cd['name_p4|addr']['mean']:,.0f} / {cd['name_p4|addr']['p99']:,.0f}" if cd else "—"),f"{100*rc['name_p4&addr']:.1f}%",(f"{cd['name_p4&addr']['mean']:.1f}" if cd else "—")])
    w(f"**{c}**"); w()
    tbl(["df ≤ T","recall name tokens","name_p4","addr tokens","recall name_p4 ∪ addr","cands mean / p99 (∪)","recall name_p4 ∧ addr","cands mean (∧)"],rows)
w("**Take-aways.** (1) Tokens are *not* rare in this corpus: a real, correct match typically shares a token with df of a few hundred to a few thousand (names are composed from a limited vocabulary; addresses from a few thousand streets/localities per country). At df ≤ 300 the union blocker recalls only 52% (US) / 65% (India) of the positives with ≈90–125 candidates; **at df ≤ 3,000 recall is 93–94% (US) / 89–91% (India) but candidates rise to ≈2,400–2,700 per S1 (p99 ≈ 7,500–8,100)**; 98.1% (US) / 95.3% (India) is reached only at df ≤ 10,000 (candidate counts were not measured there and exceed the df≤3,000 figures). (2) **Address tokens are the stronger channel** (US 86% / India 82% at df ≤ 3,000 vs name 57% / 52%), consistent with the names being noisier and 100% of non-empty matched addresses sharing ≥1 token. (3) **Name prefixes recover typos** but add little on top of the address channel. (4) An AND-blocker (name ∧ address) is tiny (≈2–3 candidates) but only reaches 39% (US) / 36% (India) recall at df ≤ 3,000 (49% / 43% if full name tokens replace the 4-char prefixes) — a useful *high-precision tier* (it contains the ‘easy’ matches), not a complete blocker. (5) Union recall converges to 100% only when every token is allowed, which is the full pool again.")
w()
w("### 11.5 Score-based retrieval: top-K by IDF-weighted token overlap (name tokens + name 4-char prefixes ×0.5 + address tokens; tokens with df>20,000 ignored)"); w()
w("Instead of a hard token rule, candidates are ranked by Σ IDF of shared tokens and the top-K kept. Evaluated on 2,400 random S1 entities per country (*sample*; 2,263 US / 2,255 India of them have ≥1 true match):")
w()
rows=[]
for c in ("US","India"):
    r=TK["recall_at_K"][c]
    for k in ("1","5","10","20","50","100"): rows.append([c,k,f"{100*r['recall_at_K'][k]:.1f}%",f"{100*r['all_matches_within_K_pct_of_entities'][k]:.1f}%",f"{1732544*int(k):,.0f}"])
tbl(["country","K","recall of true matches within top-K","entities with ALL their matches within top-K","test-scale pairs (1.73 M S1 × K)"],rows)
w("**Top-K by IDF overlap is far better than any hard rule**: K=20 already gives 93.5% (US) / 87.0% (India) match-level recall (vs ≈2,500 candidates for the df-cut union at 94%), and K=100 gives **95.7% / 90.7%**. The tail beyond K≈20 is flat (+2–4 points for 5× the candidates). The ceiling at K=100 costs ~2.8 points of macro-F0.5 even with a perfect classifier (oracle 0.9725). Recall is consistently ~5 points lower for India than for the US; consistent with (but not proven by) native-script names combined with reordered/abbreviated long addresses — this is the first thing to dissect when building the real retriever. The retrieval step in this unoptimised sparse-matrix implementation measured ≈ 45 ms (US) and ≈ 78 ms (India) per S1 entity on one core → for the 1.73 M test entities ≈ 30 core-hours single-threaded, i.e. ≈ 3 h on this 10-core machine before any optimisation (§17).")
w()
w("### 11.6 Blocking conclusions (what the data supports)"); w()
w("- **Country equality → then a *union* of retrieval channels**, because no single channel exceeds ~85% recall: (a) address-token IDF, (b) name-token/prefix IDF, (c) an exact-key tier (`sorted` address or `sorted` name equality; 47% of positives are found by one of them, §8), plus for the ~18% native-script India names an address-only channel. The channels’ union at moderate K is what reaches ≥93–95%.")
w("- **Rare name tokens**: useful, but ‘rare’ means df of a few hundred–few thousand, not tens. **Address tokens** are the most reliable signal (rare-token address channel alone: 86% recall at df ≤ 3,000). **Postal/PIN components: not available** (§7) — the closest substitutes are the numeric ‘house/plot’ token and locality tokens. **City/state tokens** are nearly useless as blockers on their own: state ≈ 1/50 of the US pool (272k candidates), and in France ~15 cities cover most records.")
w("- **Name + address combinations** are the way to get *small* candidate sets: AND tier (~2–3 candidates, ≈35–49% recall) ⊂ top-K ranked union (K≈20–50, 87–94% recall).")
w("- **Where recall is genuinely lost** (§9): empty matched address (4.4%), native-script/unrelated names, and mutated house numbers. Roughly 1% of positives have neither a usable name nor a usable address.")
w("- Scale to plan for: **K = 20–50 → 35–87 M candidate pairs for test**, K = 100 → 173 M. All are comfortable for a tree model in chunks; the retrieval step is the expensive part.")
w()
# ---------------- 12
w("## 12. SINGLETON ANALYSIS"); w()
w(f"**Singleton rate: 123,247 / 2,206,821 = {100*123247/2206821:.3f}%** (US {100*E['singleton_rate_by_country']['US']:.2f}%, India {100*E['singleton_rate_by_country']['India']:.2f}% — identical). Because macro-F0.5 gives a singleton **1.0 for predicting nothing and 0.0 for predicting anything**, this 5.6% of entities is worth 5.6 points of the final score and is the only group where a false positive costs a full 1.0.")
w()
w("**Do singletons have distinctive names/addresses?** No. Every S1-side attribute is identical for singletons and non-singletons to within sampling noise:")
w()
a=E["singleton_vs_non_S1_attrs"]
tbl(["attribute (mean)","non-singleton","singleton"],[["name length (chars)",a["name_len"]["false"],a["name_len"]["true"]],["address length (chars)",a["addr_len"]["false"],a["addr_len"]["true"]],["name tokens",a["ntok"]["false"],a["ntok"]["true"]],["# other S1 with same normalised name",a["name_mult"]["false"],a["name_mult"]["true"]],["# S1 at same normalised address",a["addr_mult"]["false"],a["addr_mult"]["true"]],["rarest name-token document frequency",a["name_min_tokdf"]["false"],a["name_min_tokdf"]["true"]],["has legal suffix",a["legal"]["false"],a["legal"]["true"]]])
w("Singleton rate by name multiplicity (1 / 2 / 3–5 / 6–20 / 21+ entities sharing the name): "+" / ".join(f"{100*v:.2f}%" for v in E["singleton_rate_by_name_multiplicity"].values())+"; by address multiplicity: "+" / ".join(f"{100*v:.2f}%" for v in E["singleton_rate_by_addr_multiplicity"].values())+"; by name length bucket: "+" / ".join(f"{100*v:.2f}%" for v in E["singleton_rate_by_name_len"].values())+". **Singletons are a uniformly random 5.6% of S1 entities**: no S1-side feature predicts them, so the decision ‘no match’ can only come from *the absence of convincing candidates*, never from the entity’s own characteristics. (Examples of real singletons look like every other entity: `Beth Presbyterian Church | 1230 Palmer Street, Downers Grove, IL`, `Ace Systems Private Limited | Lucknow, Seemant Nagar, …`.)")
w()
w("**Are they likely to create dangerous false positives? Yes — measured.** Using the realistic retrieval of §11.5 (top-100 candidates) and a small diagnostic gradient-boosting scorer (trained on 1,200 entities/country, evaluated on 1,200 held-out entities/country; *diagnostic instrument*, not the final model), the share of entities that have **at least one false candidate scored above a threshold p**:")
w()
d=TK["singleton_danger"]
tbl(["entities","n","max false-candidate p ≥ 0.5","≥ 0.8","≥ 0.9","≥ 0.95"],[["singletons (all candidates are false)",d["singletons"]["n_entities"]]+[f"{100*d['singletons'][k]:.1f}%" for k in ("P(max false p>=0.5)","P(max false p>=0.8)","P(max false p>=0.9)","P(max false p>=0.95)")],["non-singletons (their non-matching candidates)",d["non_singleton_random_B_false_cands"]["n_entities"]]+[f"{100*d['non_singleton_random_B_false_cands'][k]:.1f}%" for k in ("P(max false p>=0.5)","P(max false p>=0.8)","P(max false p>=0.9)","P(max false p>=0.95)")]])
w("So **≈ one singleton in four (24%) has a false candidate the diagnostic scorer rates ≥ 0.5, and 4.8% have one ≥ 0.95** — 1.6–2.1× the rate for non-singletons’ wrong candidates (which are diluted by having true matches to outrank them). Effect of the decision threshold on the sampled macro-F0.5 (this diagnostic scorer):")
w()
db=TK["diagnostic_baseline"]
tbl(["threshold p","macro F0.5 (random 2,400-entity sample)","US","India","singletons","non-singletons"],[[t,v["macro_F05_random_S1_sample"],v["US"],v["India"],v["singleton_score(all sampled singletons)"],v["nonsingleton_score"]] for t,v in db.items()])
w("The overall optimum is flat around p≈0.6–0.7 (0.893); raising the threshold to 0.95 lifts singleton score from 0.76 (p=0.5) to 0.95 but costs the non-singletons ~5 points. **The singleton/non-singleton trade-off is real and must be optimised explicitly on grouped validation data, not fixed at 0.5.** For scale: a false match on a 3-match entity lowers its F0.5 from 1.0 to 0.789, a missed match to 0.909 (a false positive costs ~2.3× a false negative), while a false match on a singleton costs 1.0.")
w()
w("**Where the dangerous false positives come from** (inspected: the 162 singleton false candidates with p ≥ 0.8 in the evaluation samples — 130 (80%) are unmatched decoys and 32 (20%) belong to another S1 entity):")
w("1. **Sibling look-alike records — unmatched decoys (80% of the high-scoring ones; US 68%, India 85%).** Same building/complex/street and a shared name stem but a different unit/plot number and one different name word: `Goodwill Academy | 160/218, Ajmeri Gate` vs `Goodwill Academy Exports Corp | 160/225, …`; `Ys Welfare Society | 331/9K` vs `Ys Welfare Society Private Limited | 331/12K`; `Innovative Bharat Properties Private Limited | 1/19` vs `INNOVATIVE BHARAT LOGISTICS PRIVATE LTD | 1/30`; `Kaykay Sarkar Ltd | 37/26` vs `Kaykay Sarkar Group | 37/37`; `Technologies Poineer Customer Private Limited` vs `… Public Limited`. (Only 3% of these decoys have an identical normalised name and <1% an identical address.) One cheap discriminator exists: a **conflicting legal form** (`Private` vs `Public`, `LLC` vs `Inc`) occurs in 5.4% of high-scoring false candidates vs 0.41% of true pairs (§6). These are exactly the records that a name-and-address similarity model *will* accept — and the discriminating evidence (a different plot number, a different suffix word) is the same kind of perturbation the noise generator applies to *true* matches (house number differs in ~17% of positives; suffix words are added/dropped/changed in ~35%). This is the central precision problem of the task.")
w("2. **Records belonging to *another* S1 entity (20%; US 32%, India 15%).** Multiple S1 entities at the same address (5.3% of S1 rows share an address; co-working towers, `Devya Organic Pvt. Ltd.` vs a differently named company in the same plot) — especially when the candidate’s name is in native script (`లక్ష్మీ ప్రొడ్యూసర్…`, `स्काई इंडस्ट्रीज…`), so the name cannot veto the address match. Because ground truth assigns every S2/S3 record to at most one S1 entity (§5), a one-to-one assignment step (a record goes to the S1 entity it fits best) is available to suppress this class.")
w("3. Records with an empty address are 97.8% likely to be matched to *some* entity (§13), so they are rarely decoys; their risk is instead being assigned to the wrong entity when only a short or native-script name is available.")
w("Decoy prevalence overall: 26.6% of S2 and 25.4% of S3 training records match no S1 entity, yet only **0.97% of unmatched S2 addresses occur in S1** (vs 30.4% of matched) and 5.3% of unmatched names equal an S1 name (vs 31.7% of matched) — so *most decoys are unrelated entities*, and the dangerous minority is a small, structured subset that survives blocking because it looks like the truth. **Test appears to contain proportionally more decoys** (§14), so the precision side matters more there.")
w()
# ---------------- 13
w("## 13. DATA QUALITY"); w()
dq=lambda n,c,k:P[n]["dq"][c][k]
rows=[]
for label,c,k in [("names with double space","business_name","double_space"),("names with control chars","business_name","has_control"),("addresses with control chars (0x1A / 0x7F)","business_address","has_control"),("addresses with mojibake (`Â\\x80\\x93`, `Â\\x80\\x99`)","business_address","mojibake_pat"),("addresses with HTML entity (`&#65533;`)","business_address","html_entity"),("addresses with double space","business_address","double_space"),("addresses > 150 chars","business_address","len_gt_150"),("addresses > 300 chars","business_address","len_gt_300"),("names ≤ 2 chars","business_name","len_le_2_nonempty"),("names ‘NA’/‘null’-like","business_name","literal_null_like"),("names only digits/punctuation","business_name","only_punct_digits")]:
    rows.append([label]+[dq(n,c,k) for n in ("train_source1","train_source2","train_source3","test_source1","test_source2","test_source3")])
rows.append(["leading/trailing whitespace (name or address)"]+[dq(n,"business_name","leading_trailing_ws")+dq(n,"business_address","leading_trailing_ws") for n in ("train_source1","train_source2","train_source3","test_source1","test_source2","test_source3")])
tbl(["issue (row counts)","train S1","train S2","train S3","test S1","test S2","test S3"],rows)
w("Findings:")
w()
w("- **IDs:** all 12.5 M train + 11.7 M test IDs are well-formed (`S<k>-<digits>`), **0 malformed, 0 duplicated within a file, 0 shared between train and test**; numeric parts are random 1–9-digit integers (so ID length ≈ uniform; 81.7% of matched S1/match ID pairs have equal digit-length vs 81.8% expected by chance).")
w("- **Empty strings:** the only empty field is `business_address` — S2 168,967 (3.36%), S3 175,916 (3.33%) in train; test S2 129,408 (2.65%), S3 136,098 (2.68%); S1 never. Empty-address records are almost always *matched* records (§ below), i.e. missingness is injected noise, not natural sparsity.")
w("- **Placeholder tokens inside addresses** (not empty): `NULL`, `<NULL>`, `N/A` appear in 175,568 S2 and 174,742 S3 train addresses (≈3.4%) but 79 in S1 — they must be stripped before tokenising or `null` becomes a high-df token that ‘matches’ everything.")
w("- **Pandas default NA parsing corrupts data:** names such as `NA`, `Na`, `NAN`, `null`, and any address `N/A`/`NULL` would silently become `NaN`; reading with `keep_default_na=False, dtype=str` avoids this. Use `quoting=csv.QUOTE_NONE` (stray `\"` in 4–349 rows/file).")
w("- **Control characters / encoding damage (small but real, and present in S1 too, i.e. from the source data rather than the noise generator):** `\\x1A` (SUB) has replaced apostrophes/special characters (`Shopper\\x1aS Stop`, `D\\x1asouza Colony`, name `Medchal\\x1aMalkajgiri Housekeeping Private Limited` appears 1/3/1 times in S1/S2/S3 — the same entity), 70/119/125 train addresses; UTF-8 en-dash/apostrophe mis-decoded as Latin-1 (`Â\\x80\\x93`, `Â\\x80\\x99`) in 443/1,118/785 train addresses (0.02% of rows; these are the U+0080–U+0099 ‘Cc’ characters); `&#65533;` numeric HTML entity for the replacement char. There are **no U+FFFD characters, no BOMs, no tabs/newlines inside fields, and all non-ASCII names in a 30k-name sample were already NFC**; in a 200k-row sample per file the only odd characters of note are ZERO WIDTH NON-JOINER (U+200C, inside Indic conjuncts), `°` (degree sign in addresses), typographic apostrophe U+2019, and the C1 controls U+0080–U+0099 from the mojibake.")
w("- **Extremely long fields:** names ≤ 123 chars; addresses ≤ 269 chars (p99.9 ≈ 150); 0.09% of S1 and 0.04–0.06% of S2/S3 addresses exceed 150 chars (all Indian multi-line addresses with repeated district/locality names, e.g. `Visakhapatnam, Visakhapatnam, Visakhapatnam, Vishakhapatnam`). Nothing >300.")
w("- **Very short / junk names:** S3 has 9,275 names of ≤2 characters in train (`SC`, `SI`, `SS`, `GM` …; mostly India 7,799), S2 only 704. **In test the pattern moves to France:** test S2 has 10,166 (9,748 France = 1.39% of France S2 rows vs 0.014% of train S2) and S3 19,910 (12,182 France). These 2-letter names look like initials of the real name (`CC`, `AF`, `BG`) — **no name evidence at all** — and are a new, France-specific S2 noise type absent from train (§14). Also 194–423 names are only digits/punctuation (`18  29`, `1 800`).")
w("- **Suspiciously repeated names/addresses:** in S1 45–46% (India) / 36% (US) of rows share a name with ≥1 other S1 row (max 253 for `Primary Care Group`); 5% share an address. S2/S3 top repeated names are the generic `Primary Care`, `Physical Therapy`, `Womens Health`, `earnosethroat.com`, `CC`. Shared addresses in S3 India include truncated addresses like `Ground Floor, Bangalore, KA` (29 records), `303, Mumbai, MH` — addresses stripped to a floor/door number and a city, which are *not identifying*.")
w(f"- **Exact duplicate rows inside S2/S3:** S2 has {DU['S2']['rows']:,} rows in {DU['S2']['dup_groups']:,} groups of identical (name, address, country); S3 {DU['S3']['rows']:,} rows / {DU['S3']['dup_groups']:,} groups. **100% of these rows are matched, and in 100% of the groups all rows are matched to the same S1 entity** (0 mixed, 0 unmatched). An exact in-source duplicate is therefore a certain positive-cluster signal (a free consistency constraint, and a leakage-like structure to be aware of).")
w("- **Record-level artefacts that predict ‘matched’ (measured on train S2; S3 is the same):**")
w()
pr=T["matched_vs_unmatched_priors"]["S2"]
tbl(["record feature","unmatched (decoy) S2","matched S2","P(matched | feature)"],[
 ["address empty",f"{100*pr['addr_empty']['false']:.2f}%",f"{100*pr['addr_empty']['true']:.2f}%","**97.8%**"],
 ["name contains `.com`",f"{100*pr['dotcom']['false']:.2f}%",f"{100*pr['dotcom']['true']:.2f}%","**≈ 96%**"],
 ["mean name length (chars)",f"{pr['name_len']['false']:.1f}",f"{pr['name_len']['true']:.1f}","shorter ⇒ more likely matched"],
 ["mean name tokens",f"{pr['name_ntokens']['false']:.2f}",f"{pr['name_ntokens']['true']:.2f}",""],
 ["name ALL-CAPS",f"{100*pr['name_upper']['false']:.1f}%",f"{100*pr['name_upper']['true']:.1f}%",""],
 ["country = US",f"{100*pr['country_US']['false']:.1f}%",f"{100*pr['country_US']['true']:.1f}%","no difference"],
 ["ID length / ID value / row position (KS distance)","—","—","KS 0.005 / 0.005 / 0.005 → no signal"]])
w("Decoys are *cleaner and longer* than matched records (matched records have been through the noise generator: truncated names, `.com` forms, blanked addresses; decoys were generated as complete standalone entities). This is a legitimate record-level feature at test time, but it (a) inflates any validation that treats ‘noisy = match’, and (b) may not transfer to France, where the noise mixture differs (initials-only names).")
w()
# ---------------- 14
w("## 14. TRAIN / TEST COMPARISON"); w()
def g(n,k,sub=None):
    v=P[n][k]; return v[sub] if sub else v
rows=[]
for k in (1,2,3):
    a,b=f"train_source{k}",f"test_source{k}"
    rows.append([f"S{k} rows",f"{g(a,'rows'):,}",f"{g(b,'rows'):,}",f"{g(b,'rows')/g(a,'rows'):.2f}×"])
rows.append(["S2 records per S1 entity",f"{g('train_source2','rows')/g('train_source1','rows'):.2f}",f"{g('test_source2','rows')/g('test_source1','rows'):.2f}",f"{(g('test_source2','rows')/g('test_source1','rows'))/(g('train_source2','rows')/g('train_source1','rows')):.2f}×"])
rows.append(["S3 records per S1 entity",f"{g('train_source3','rows')/g('train_source1','rows'):.2f}",f"{g('test_source3','rows')/g('test_source1','rows'):.2f}",f"{(g('test_source3','rows')/g('test_source1','rows'))/(g('train_source3','rows')/g('train_source1','rows')):.2f}×"])
for k in (1,2,3):
    a,b=f"train_source{k}",f"test_source{k}"
    rows.append([f"S{k} name length mean / p95 / max",f"{g(a,'name_len','mean')} / {g(a,'name_len','p95')} / {g(a,'name_len','max')}",f"{g(b,'name_len','mean')} / {g(b,'name_len','p95')} / {g(b,'name_len','max')}",""])
for k in (1,2,3):
    a,b=f"train_source{k}",f"test_source{k}"
    rows.append([f"S{k} address length mean / p95 / max",f"{g(a,'addr_len','mean')} / {g(a,'addr_len','p95')} / {g(a,'addr_len','max')}",f"{g(b,'addr_len','mean')} / {g(b,'addr_len','p95')} / {g(b,'addr_len','max')}",""])
for k in (2,3):
    a,b=f"train_source{k}",f"test_source{k}"
    rows.append([f"S{k} empty address",f"{g(a,'cols_detail')['business_address']['missing_pct']:.2f}%",f"{g(b,'cols_detail')['business_address']['missing_pct']:.2f}%",""])
for k in (1,2,3):
    a,b=f"train_source{k}",f"test_source{k}"
    rows.append([f"S{k} names non-ASCII",f"{100*g(a,'dq')['business_name']['non_ascii_rows']/g(a,'rows'):.1f}%",f"{100*g(b,'dq')['business_name']['non_ascii_rows']/g(b,'rows'):.1f}%",""])
    rows.append([f"S{k} names with double space",f"{100*g(a,'dq')['business_name']['double_space']/g(a,'rows'):.1f}%",f"{100*g(b,'dq')['business_name']['double_space']/g(b,'rows'):.1f}%",""])
rows.append(["mean address tokens S1 / S2 / S3",f"{g('train_source1','addr_tokens')} / {g('train_source2','addr_tokens')} / {g('train_source3','addr_tokens')}",f"{g('test_source1','addr_tokens')} / {g('test_source2','addr_tokens')} / {g('test_source3','addr_tokens')}",""])
tbl(["statistic","train","test","test/train"],rows)
w("**Countries:** train US 60.0% / India 40.0%; test US 38.3% / India 46.8% / **France 15.0%** of S1 (14.4% of S2/S3). Per-country S1 structure for the two shared countries is **essentially identical** to train — India 45.3% of S1 rows share a name (train 46.0%), US 29.2% (train 35.9% — the lower US share is the expected size effect of a smaller S1 pool: 663k vs 1.32M entities); India name length 26.4 (26.4), address 77.7 chars (77.7), 11.2 tokens (11.2); US name 22.5 (22.5), address 35.0 (35.0), 5.9 tokens (5.9); legal-suffix mix identical (India `Private Limited` 73.1% vs 73.3%; US `LLC` 50.2% vs 50.0%, `Incorporated` 33.8% vs 33.8%). The generator is unchanged for US/India; the shift is (a) the *mix* of countries, (b) France, (c) a larger S2/S3 pool relative to S1.")
w()
w("**Distribution shift to pay attention to**")
w()
w("1. **France is an unseen domain, ≈15% of test entities** (≈260k S1 entities; 1.43 M S2/S3 records). 53% of French S1 name tokens and 37% of French address tokens never occur in any training file (vs 1.1–1.5% for the same measurement on US/India test names). French names are short (mean 19.4 vs 24 chars), template-like (`Bordeaux Club SARL` appears 205 times as an S1 name; `Nantes Club SARL` 157) and 34% of French S1 rows share their name with another entity; **8.5% share their address (train ≈5%; up to 99 entities at `12 RUE Lyderic, Lille`)**, so duplicates-with-different-entity (the twin problem) is likely *worse* in France. There are 15.7% accented names in S1 (train S1: 0%), French legal forms (`SARL/SAS/EURL/SASU/SCI/SA`, S2/S3 also `S.A.S`, `S.A.R.L`), abbreviated street types (`R.`, `Av`, `Bd`, `Pl.`, and `Allée`/`Allee` spellings), no postal codes, and a small set of ~15 cities — geography is uninformative.")
w("2. **Larger decoy share in test.** Test has 2.82 S2 and 2.93 S3 records per S1 entity vs 2.28 / 2.39 in train (+24% / +23%). Independent check: the share of S2/S3 records whose (token-sorted) address also occurs in S1 is **~20% lower in test** at the same country (S2 US 23.6% vs 29.3%; S2 India 9.4% vs 11.9%; S3 US 4.0% vs 4.9%; S3 India 3.6% vs 4.5% — a uniform ≈0.80× factor), while the share of *S1* entities that have an exact-address hit in S2∪S3 is unchanged (US 46.9% vs 47.5%; India 29.3% vs 29.5%). Both point the same way: **similar matches per S1 entity, but ≈ 41% of test S2/S3 records unmatched vs ≈ 26% in train** (solving `share = m·a + (1−m)·0.0097` with the train-calibrated matched-record rate `a`, per country, gives m ≈ 0.59 for both US and India) (inference from unsupervised proxies, not a label; if true, the false-positive pressure per S1 entity is higher in test and precision-oriented thresholds tuned on train will be slightly too permissive).")
w("3. **New noise type in France S2:** 1.4% of France S2 records (and 1.7% of France S3) have 2-letter, initials-like names vs 0.014% in train S2. Name uppercase rates differ too (France S2 names all-caps 20.8%, S3 5.6% — same style split as train).")
w("4. **No entity/ID leakage between splits:** 0 shared IDs; 0 test S1 rows whose (normalised name, address) pair exists in train S1. Names *are* reused across the split (35% of test S1 normalised names occur in train S1: India 46%, US 36%, France 0.01%) because names are template-generated, so name-frequency features fitted on train will partly transfer for US/India and not at all for France; addresses rarely repeat (5% of test S1 addresses occur in train S1).")
w("5. **Vocabulary:** US/India test tokens are only slightly more OOV than a held-out train baseline (name tokens 1.5% vs 1.15% US, 1.1% vs 0.75% India; address tokens 1.9% vs 1.5% US, 2.9% vs 2.1% India) — i.e. no vocabulary shift for the shared countries beyond the ~+0.4–0.8-point out-of-sample effect.")
w()
open("report_C.md","w").write("\n".join(out)); print(len(out))
