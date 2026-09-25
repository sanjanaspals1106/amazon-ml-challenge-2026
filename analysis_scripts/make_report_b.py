import json, pandas as pd, numpy as np
exec(open("make_report_a.py").read().split("names=[")[0].split("out=[]")[0])  # imports/loads only
P=json.load(open("profile.json")); G=json.load(open("gt_stats.json")); X=json.load(open("out/exact_summary.json")); NZ=json.load(open("out/noise.json"))
SIM=json.load(open("out/similarity.json")); T=json.load(open("out/traintest.json")); AP=json.load(open("out/addr_parts.json"))
out=[]
def w(s=""): out.append(s)
def esc(x,n=70):
    x=str(x).replace("|","\\|").replace("\n"," ").replace("\t"," ").replace("\x1a","<0x1A>")
    return (x[:n]+"…") if len(x)>n else x
def tbl(h,rows):
    w("| "+" | ".join(h)+" |"); w("|"+"|".join(["---"]*len(h))+"|")
    for r in rows: w("| "+" | ".join(esc(c,110) if isinstance(c,str) else (f"{c:,}" if isinstance(c,int) else str(c)) for c in r)+" |")
    w()
def ex(lst,n=3,sep="  →  "): return "<br>".join(f"`{esc(a,60)}`{sep}`{esc(b,60)}`" for a,b,*_ in lst[:n])
# ---------------- 6
w("## 6. NAME NOISE ANALYSIS"); w()
w("Method: 500,000 random positive pairs *(sample)*; the S1 name is compared with the matched S2/S3 name. Categories in the first table are **mutually exclusive and priority-ordered** (first rule that fires): identical → case-only → whitespace/punctuation-only → abbreviation/diacritic/`&`-only (`ext` equal) → word-order-only → native-script rewrite → DBA/FKA alias → extra tokens (S1 tokens ⊂ match) → dropped tokens (match ⊂ S1) → otherwise by `token_sort_ratio` band. Because of the priority order, compound noise (typo + suffix change + case change) falls into the lower buckets. The second table lists *independent* pattern flags, which overlap.")
w()
rows=[]
LBL={"typo/mixed_edits(sim>=80)":"mixed edits incl. typos, punctuation inside suffix, etc. (token-sort sim ≥ 80)","heavy_edits(50<=sim<80)":"heavy edits incl. domain-style names, honorifics, alias words (50 ≤ sim < 80)","unrelated_or_alias(sim<50)":"unrelated alias / gibberish / initials (sim < 50)","dba/fka/ta_alias":"dba / f/k/a / t/a alias","extra_tokens_in_match(suffix/honorific/dup)":"extra tokens in match (suffix, honorific, duplicated word)","dropped_tokens_in_match":"dropped tokens in match","native_script_translit":"native-script rewrite","abbrev/diacritic/&_only":"abbreviation / diacritic / `&`-`and` only","whitespace/punct_only":"whitespace / punctuation only","case_only":"case only","word_order_only":"word order only","identical":"identical"}
for c,v in NZ["name_taxonomy"].items(): rows.append([LBL.get(c,c),f"{v['pct']:.2f}%",f"{v['pct_S2']:.1f}%",f"{v['pct_S3']:.1f}%",f"{v['pct_US']:.1f}%",f"{v['pct_India']:.1f}%",ex(v["examples"],2)])
tbl(["primary relation of matched name to S1 name","all","S2","S3","US","India","real examples (S1 → matched)"],rows)
w("**Key observation:** only **4.6%** of true pairs have byte-identical names; 10.75% are equal after lowercasing; 21.9% after `basic`; 34.6% even after `ext`+token-sorting (§8). Roughly two thirds of positives carry *real* string differences that exact/blocked-equality logic cannot resolve.")
w()
f1,f2,f3=NZ["name_flags_S1side"],NZ["name_flags_matchside_S2"],NZ["name_flags_matchside_S3"]
keys=["all_upper","all_lower","double_space","leading_junk_punct","hash_or_at_prefix","honorific_prefix","id_suffix","has_dba_fka_ta","dot_com_domain","leet_or_digit_in_word","non_ascii_latin","native_script","duplicate_adjacent_token","parenthesized_or_bracketed","ampersand","word_and","plus_sign","single_token","legal_suffix_present"]
tbl(["pattern (% of names)","S1 side (clean reference)","matched S2 name","matched S3 name"],[[k,f1[k],f2[k],f3[k]] for k in keys])
w("**S1 names are the clean canonical form** (no double spaces, no all-caps, no accents, no aliases; ≈67% carry a legal suffix). **All the noise is injected into S2/S3**, with source-specific styles: S2 renders 20.6% of names in ALL CAPS (S3 3%) and has no DBA aliases, while S3 has 3.5% `X dba/fka/t/a <S1 name>` aliases and S3’s US addresses spell out the state name (§7).")
w()
w("### Legal-suffix behaviour"); w()
ls=NZ["legal_suffix"]
w(f"Legal-token set identical to S1’s in {ls['same_set']}% of pairs; the match **drops** a suffix that S1 has in {ls['match_has_none_S1_has']}%; the match **adds** a suffix S1 lacks in {ls['match_has_S1_none']}%. Most frequent transitions (S1 legal tokens → matched-name legal tokens; `-` = none) among the pairs where they differ:")
w()
tbl(["transition","count (in 500k sample)"],[[a,b] for a,b in NZ["legal_transitions_top"][:14]])
SW=json.load(open("out/suffix_swaps.json"))
w("**Which suffix swaps are real?** Measured on 1,000,000 random true pairs: for each abbreviation/full-form couple, the rate at which S1 uses one form and the matched name uses the other *(and not the S1 form)*:")
w()
tbl(["couple","S1 has abbreviation only → match has full form","S1 has full form only → match has abbreviation","S1 names with abbrev-only / full-only (of 1 M pairs)"],[[f"`{r['abbr']}` ↔ `{r['full']}`",f"{r['swap_rate_when_S1_abbr_%']}%",f"{r['swap_rate_when_S1_full_%']}%",f"{r['S1_has_abbr_only']:,} / {r['S1_has_full_only']:,}"] for r in SW])
w("**Findings.** (a) `Ltd`/`Limited`, `Corp`/`Corporation`, `Co`/`Company` swap in *both* directions at ≈ 5–11% per occurrence; `Inc → Incorporated` swaps at 5.0% but only in that direction (S1 itself almost never contains `Incorporated`: 2 of 1 M pairs). (b) **`Pvt` ↔ `Private` never flips: 0 swaps in 1,000,000 pairs** (a single case, `Acme Solid Private Limited → Acme Solid Pvt Limited`, was found in a separate 600k-pair scan) — even though the task description lists it. S1 itself uses both forms as separate, stable styles (India S1: `Private Limited` 48.9% of names, `Pvt Ltd` 13.8%) and the matched record keeps S1’s choice (`Private Limited → Private Ltd`, `Private`, `Limited`, dropped; `Pvt Ltd → Pvt Limited`, `Pvt`, dropped, `[PVT]`). (c) In the US `LLC → L.L.C.`, `Inc → Incorporated / Inc.`, `Corp → Corporation`. (d) **A conflicting legal form is almost never noise:** counting pairs where both names carry legal tokens and neither token set contains the other (e.g. `Private` vs `Public`, `LLC` vs `Inc`), true pairs show 0.41% (n=400k) vs **5.4% (n=934) of high-scoring false candidates (p ≥ 0.5) — a ≈13× enrichment** — and 5.9% of true pairs merely differ in some legal token (`Ltd` vs `Limited`). **Dropping/adding a suffix is weak evidence and must not be penalised (dropping is the most common difference, 22%), but a *conflicting* suffix is a strong negative signal.**")
w()
w("### Representative examples by noise type (real S1 → S2/S3 pairs)"); w()
lab={"Pvt<->Private":"Pvt ↔ Private","Ltd<->Limited":"Ltd ↔ Limited","Corp<->Corporation":"Corp ↔ Corporation","Inc<->Incorporated":"Inc ↔ Incorporated","Co<->Company":"Co ↔ Company","&<->and":"& ↔ and","LLC<->L.L.C.":"LLC ↔ L.L.C. / punctuation in legal form","typo(sim 80-95, same token count)":"typos / character edits","leet/digit substitution":"digit-for-letter (leet) substitution","diacritic injection (S1 ascii)":"case/diacritic injection (S1 is ASCII)","domain-form name":"domain-style name (`xxx.com`, `www.`)","honorific added":"honorific / prefix added (Mr, Sri, Shri, Dr, M/s)","(ID: n)/#n suffix":"fake ID suffix `(ID: 12345)` / `#12345`","bracketed legal token":"bracketed legal token `[Ltd]`, `(LLC)`","leading junk punctuation":"junk prefix (`...`, `--`, `>>`, `[P.C.]`)","duplicated token":"duplicated token","transliteration to native script":"transliteration/translation to native script","DBA/FKA/TA":"DBA / F/K/A / T/A alias (alias name is random, S1 name is inside)","unrelated alias (sim<30, latin)":"unrelated alias / gibberish or 1–2 letter acronym"}
AB=json.load(open("out/abbrev_examples.json"))
rows=[]
for k in lab:
    if k not in NZ["name_examples"]: continue
    if k=="Pvt<->Private": rows.append([lab[k],"**no true pair found** (0 in 1 M sampled pairs; see the swap-rate table above) — S1 `Pvt Ltd` stays `Pvt Ltd`/`Pvt`, S1 `Private Limited` stays `Private …`"]); continue
    rows.append([lab[k],ex(AB[k],3) if k in AB and AB[k] else ex(NZ["name_examples"][k],3)])
tbl(["noise type","examples (S1 name → matched name)"],rows)
w("Additional name-noise facts *(sample)*: "
  f"(i) **word-order changes** are 6.0% as the *only* difference (`Koch and Hartsfield Kensington LLC → Koch LLC Hartsfield Kensington and`) and are also embedded in many of the ‘typo/mixed’ pairs; the legal suffix and `and` are frequently moved into the middle. "
  f"(ii) **Typos** are keyboard-agnostic random character edits (insert/delete/substitute/transposition; `Keystone Drive → Keystone Dnaige`, `Metropolitan Optics → Metropolitan Otpamcs`) plus digit-for-letter swaps (`El0ise`, `6hazipur`, `lnc` for `Inc`, `5ERVICES`) that appear in ~2.3% of matched names. "
  f"(iii) **Transliteration**: 9.3% of S2 and 5.3% of S3 matched names are rewritten *entirely in native script* (Devanagari dominant; also Telugu, Kannada, Bengali, Tamil, Gujarati, Malayalam, Gurmukhi, Oriya), i.e. 18% of all India pairs; Latin token overlap with S1 is zero for these, so they can only be matched through the address or through a learned transliteration. They are not literal transliterations of rare words but of the *whole* name, e.g. `Hitech Agro Private Limited → हाईटेक एग्रो प्राइवेट लिमिटेड`. "
  f"(iv) **DBA/alias patterns** ({NZ['dba_n']:,} in the 500k sample, all in S3): markers `dba`, `formerly`, `f/k/a`, `aka`, `d/b/a`, `t/a`, `trading as`, `fka`, `a/k/a`; in {NZ['dba_contains_s1name_pct']}% of them the true S1 name appears verbatim after the marker (`Veraarcevo+ fka American School of Medicine LLC`), while the prefix is a meaningless invented brand. "
  f"(v) **Unrelated aliases**: {NZ['name_taxonomy']['unrelated_or_alias(sim<50)']['pct']}% of true pairs have `token_sort_ratio`<50 to the S1 name and are not native-script (`Commercial Electronics International LLC → Calomira`, `IT`, `Zetayuma`, `Gildlumxylo`) — for these the name carries **no** information and only the address can match. "
  f"(vi) **Extra/duplicated tokens** (`Pediatric Pediatric LLC Center`, `Smith LLC Center`) and generic added words (`Center`, `Services`, `Partners`, `Company`) appear on ~10% of pairs; **dropped tokens** on ~15% (`Behavioral Health Medicine of Washington Inc → … Washington`).")
w()
w("Name-similarity distribution of true pairs (`token_sort_ratio` on `ext`-normalised names): p5 = 10.5, p10 = 42.6, p25 = 72.7, median = 89.4, p75 = 100 — i.e. the lower quartile of true pairs looks like a non-match by name.")
w()
# ---------------- 7
w("## 7. ADDRESS NOISE ANALYSIS"); w()
w("Same 500k-pair *(sample)* and the same priority-ordered scheme. Empty matched address is its own category. `partial_overlap` = token-Jaccard ≥ 0.5 after `ext`; `weak_overlap` = 0 < Jaccard < 0.5.")
w()
rows=[]
for c,v in NZ["addr_taxonomy"].items(): rows.append([c,f"{v['pct']:.2f}%",f"{v['pct_S2']:.1f}%",f"{v['pct_S3']:.1f}%",f"{v['pct_US']:.1f}%",f"{v['pct_India']:.1f}%",ex(v["examples"],2)])
tbl(["primary relation of matched address to S1 address","all","S2","S3","US","India","real examples (S1 → matched)"],rows)
w("**Key observations.** (a) Only 2.2% of matched addresses are byte-identical (4.3% of S3 pairs; **0% of S2 pairs**, whose addresses are always re-cased/re-formatted) and 8.3% are equal after `basic`. (b) **The two sources apply different transformations** (state formats measured on 400k-row samples). *Both* S2 and S3 abbreviate street types (~45–49% of US addresses), insert `NULL`/`N/A` fillers (~3–4%), zero-pad house numbers (~4.5–4.9%), add `PO BOX`/`PMB` (~1.6–1.7% of US) and `#`/unit tokens. **S2** additionally is ALL-CAPS (93% of US S2 addresses; 25% of India), keeps the US state as a 2-letter code (like S1) and the India state as a full Latin name (77%) or native script (21%), shuffles component order (8.4% reorder-only vs 1.9% in S3) and drops components (17% ‘subset of S1’ vs 0.3% in S3). **S3** is mixed-case, **spells out the US state** (95% of US S3 addresses; `Texas` vs S1 `TX`), writes the **India state as a 2-letter code** (~70% of India S3 addresses: `GJ`, `UP`, `TN`) or in native script (~20%), and substitutes/adds districts and cities (54.6% partial overlap, 23.8% weak overlap vs 26.4%/4.9% in S2). A state/city normaliser therefore needs code↔name↔native-script maps (learnable from the training pairs). (c) Address token-Jaccard between true pairs (non-empty): p5 0.30, p25 0.56, median 0.71, p75 0.87 — high but far from 1. (d) A true pair with **zero** shared address token is essentially nonexistent (0.005% of non-empty pairs) — *every non-empty matched address still shares at least one token with the S1 address*, which is why address tokens are such a strong blocking signal (§11).")
w()
af=NZ["addr_flags"]
keys=["all_upper","abbr_Rd/St/Ave/Dr/Ln/Ct/Cir/Blvd/Hwy","full_Road/Street/Avenue/Drive/Lane/Court","US_full_state_name_at_end","NULL/N/A/<NULL>_placeholder","PO_BOX","PMB","unit_or_#","leading_zero_number","leading_hash","fraction_slash_1/2","suffix_letter_after_number(3707-D)","landmark(near/opp/behind/beside/next to)","native_script","double_space"]
tbl(["pattern (% of non-empty addresses)","S1 US","S2 US","S3 US","S1 India","S2 India","S3 India"],[[k,af["S1side_US"][k],af["S2side_US"][k],af["S3side_US"][k],af["S1side_India"][k],af["S2side_India"][k],af["S3side_India"][k]] for k in keys])
w("Reading the table: **Rd/Road, St/Street** — S1 US spells out 87% of street types (`Road`, `Street`), S2/S3 abbreviate ~45–49% (`RD`, `ST`, `AVE`) → an abbreviation dictionary is worthwhile. **Landmarks** (`Near/Opp/Behind/Beside`) are an India phenomenon (11.5% of S1-India addresses, 0.01% in US) and appear intact in S2 (`NEAR MDSD GIRLS COLLEGE`) but are *sometimes dropped* in S3 (7.7% of S3-India addresses contain a landmark keyword vs 11.5% in S1; e.g. `Plt No. 6, Chandrapur, Gadchandur, MH`). **Native script** appears in 22–24% of India S2/S3 addresses, almost always only for the *state* (`महाराष्ट्र`, `తెలంగాణ`, `ગુજરાત`, `தமிழ்நாடு`) and sometimes for the whole tail; US has none. **Municipal numbering** (`6-3-907/912/C`, `H.no 910 A 3503`, `1-1-770/A`, `C-46`) is heavily used in India and is preserved verbatim in most variants (the first number is unchanged in 79% of pairs, see below), making it a highly discriminative substring, but S3/S2 sometimes prepend/alter it (`No 85 67`, `HN A-834`, `#449`, `DORO NO 0689`).")
w()
w("### Missing / altered components — measured *(sample: 400k US pairs per source)*"); w()
a2,a3=AP["US_S2"],AP["US_S3"]
tbl(["US pairs","S2 match","S3 match"],[["S1 address has a state","100%","100%"],["**state missing** in match (when S1 has it)",f"{a2['match_missing_state_when_S1_has_pct']}%",f"{a3['match_missing_state_when_S1_has_pct']}%"],["S1 city found verbatim in match",f"{a2['S1_city_found_exact_in_match_pct']}%",f"{a3['S1_city_found_exact_in_match_pct']}%"],["**city missing or misspelled** (fuzzy≥85 fails)",f"{a2['S1_city_missing_or_misspelled_in_match_(fuzzy85)_pct']}%",f"{a3['S1_city_missing_or_misspelled_in_match_(fuzzy85)_pct']}%"],["city present but misspelled (`BALCH SPRINS`, `Clolierville`)",f"{a2['city_present_but_misspelled_pct']}%",f"{a3['city_present_but_misspelled_pct']}%"],["street number present in match (S1: 100%)",f"{a2['match_has_street_number_pct']}%",f"{a3['match_has_street_number_pct']}%"],["ZIP code present (either side)","0%","0%"]])
w("**PIN / postal codes are absent from the data.** No source, in either train or test, contains ZIP codes (0.000% of US addresses end in `ST 12345`), Indian PIN codes (6-digit tokens appear in only 0.3–0.8% of addresses and are plot/survey numbers, not PINs), or French code postaux (`59000 Lille` pattern ≈ 0%). *There is no postal-code blocking key.* The geographic signals available are state, city, district/locality strings and the house/plot number.")
w()
hn=NZ["house_number"]
w(f"**House / plot number** (first numeric token; both addresses have one in {hn['both_have_number_pct']}% of pairs): equal in {hn['first_number_equal_pct_of_both']}% of those pairs, {hn['equal_after_strip_leading_zeros_pct_of_both']}% after stripping leading zeros (US {hn['first_number_equal_US']}%, India {hn['first_number_equal_India']}%). So the number is wrong or different in ~17% of true pairs: the noise process *mutates* numbers as well (`3979 → 979`, `229 → 82`, `128 → 28`, `13510 → 013510`, `73 → 73 1/2`, `3707 → 3707-D`). A house-number blocking key would lose ~1 in 6 true pairs.")
w()
tbl(["address noise type","real examples (S1 → matched)"],[[k,ex(v,3)] for k,v in NZ["addr_examples"].items() if len(v)])
w("Transliteration variants of place names occur in India (`Ahmadabad`/`Ahmedabad`, `Kolkta`, `Poona`/`Pune`, `Madras`/`Chennai`, `Bangalore`/`Bengaluru`, `Eranakulam`/`Ernakulam`, `Keralam`/`Kerala`, `Vishakhapatnam`/`Visakhapatnam`, `Gautam Budh Nagar`/`Gautam Buddha Nagar`), and *city substitution* to a neighbouring district (`Rajkot → Porbandar`, `New Delhi → Gautam Buddha Nagar`, `Phoenix → NORTHEAST` (AZ), `Toledo → ‘Tledo’`) means even the city is not reliable. Long Indian addresses (p95 = 103 chars, up to 268) carry repeated sub-localities that are shuffled or dropped.")
w()
# ---------------- 8
w("## 8. EXACT MATCH ANALYSIS (all 7,638,365 training positives)"); w()
w("Computed on **every** true pair (not a sample), with the normalisations defined at the top. ‘Address’ figures require a non-empty matched address (4.4% of positives have an empty address and can never match on it).")
w()
tbl(["normalisation","name equal","address equal","BOTH equal","name OR address equal","NEITHER equal"],[
 ["raw (byte-identical)",f"{X['name_raw']}%",f"{X['addr_raw']}%","—","—","—"],
 ["lowercase only",f"{X['name_lower']}%","—","—","—","—"],
 ["**basic** (lower + punct→space + whitespace)",f"{X['name_basic']}%",f"{X['addr_basic']}%",f"{X['both_basic']}%",f"{X['either_basic']}%",f"**{X['neither_basic']}%**"],
 ["**ext** (basic + `&`→and + diacritic fold + abbreviations)",f"{X['name_ext']}%",f"{X['addr_ext']}%",f"{X['both_ext']}%",f"{X['either_ext']}%",f"**{X['neither_ext']}%**"],
 ["**sorted** (ext + token-sorted → order-invariant)",f"{X['name_sorted']}%",f"{X['addr_sorted']}%",f"{X['both_sorted']}%",f"{X['either_sorted']}%",f"**{X['neither_sorted']}%**"]])
w(f"Name-only equal (basic): {X['name_only_basic']}%; address-only equal (basic): {X['addr_only_basic']}%. Address is empty for {X['addr_empty']}% of positives; the matched record is in the same country for {X['country_same']}%.")
w()
w("**Interpretation.** Even with the most generous deterministic normalisation, **53% of true matches share neither an identical name nor an identical address** (71% under plain `basic`), and only 5% share both. Exact-key joins can therefore recover at best ~47% of matches (the rest need fuzzy comparison) — while by construction they are the safest matches. Breakdown of the `basic` numbers: by source — name 21.4% (S2) / 22.2% (S3), address 12.5% (S2) / **4.3% (S3)** (S3 formats addresses more differently: full state names, dropped components); under `sorted`, S2 address equality reaches 28.9% but S3 only 6.2%. By country (S1 side) — name basic-equal India 15.9% / US 25.8%; address `sorted`-equal India 10.5% / US 21.6%. India is harder for exact logic because of native-script names, longer multi-component addresses and heavier number/locality perturbation.")
w()
w("**Exact matches are *not* automatically unambiguous.** Because 38% of S1 rows share their name with another S1 entity (§3), an exact-name hit has to be disambiguated by address: in the hard-negative test of §9, an S1 name paired with the match of a *different* S1 entity that has the identical name gets `token_sort_ratio` ≈ 0.72 on average and the name features cannot separate them (AUC ≈ 0.55).")
w()
open("report_B1.md","w").write("\n".join(out)); print(len(out))
