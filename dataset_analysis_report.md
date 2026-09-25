# Amazon ML Challenge 2026 — Business Entity Resolution: Dataset Reconnaissance Report

**Scope:** reconnaissance only. No final model was trained, the original dataset was not modified (parquet/JSON caches and scripts live outside `dataset/`), and no external data, APIs, geocoders or web lookups were used. Everything below is measured from the supplied files.

**Environment:** macOS, 10 cores, 16 GB RAM, Python 3.13, pandas 2.2, scikit-learn 1.6, rapidfuzz 3.14, scipy. Files were read with `sep="\t"`, `quoting=csv.QUOTE_NONE`, `dtype=str`, `keep_default_na=False` (see §13 for why the last two matter).

**How to read the numbers.** Full-file statistics (sizes, schema, duplicates, missingness, ground-truth structure, exact-match rates over all 7.64 M positive pairs) are exact. Statistics marked *(sample)* come from random samples of pairs/entities (size stated) — the sampling error is small for the headline rates but not for rare categories. Blocking candidate counts come from 1,500-entity samples per country; the ‘diagnostic’ baseline in §11/§12 uses 1,200 training and 1,200 evaluation entities per country and is a **measurement instrument, not a proposed final model**.

**Normalizations used throughout** (all deterministic, no external resources):

- `basic`: lowercase → every ASCII punctuation/symbol char (and common Unicode quotes/dashes/danda `।`) replaced by a space → whitespace collapsed → strip. Accents and native scripts are kept.
- `ext` (extended): `basic` + `&`→`and` + Latin-diacritic folding (NFKD, drop U+0300–036F only, so Indic vowel signs are untouched) + abbreviation expansion (`pvt→private, ltd→limited, corp→corporation, inc→incorporated, co→company`; for addresses also `rd→road, st→street, ave/av→avenue, dr→drive, ln→lane, ct→court, cir→circle, blvd/bd→boulevard, hwy→highway, ter→terrace, pl→place, apt→apartment, fl→floor, nr→near, opp→opposite, r→rue, n/s/e/w→compass` …) + for addresses, `NULL`/`N/A` placeholders dropped. `sorted` = `ext` with tokens sorted (removes word/component order).

Files produced next to this report: `dataset_analysis_report.md`, `dataset_analysis_summary.json`, and `analysis_scripts/` (the main pipeline scripts `s00`–`s11` plus the report builders; a few one-off checks — suffix-swap rates, the legal-form-conflict rate, the Unicode/control-character scan, the singleton-decoy inspection — were run interactively and are **not** saved as scripts).

## 1. FILE SIZES

| file | size | data rows (excl. header) | columns |
|---|---|---|---|
| train/train_source1.tsv | 210,069,713 B (200.3 MiB) | 2,206,821 | 4 |
| train/train_source2.tsv | 489,301,488 B (466.6 MiB) | 5,034,616 | 4 |
| train/train_source3.tsv | 503,705,637 B (480.4 MiB) | 5,285,603 | 4 |
| train/train_ground_truth.tsv | 127,015,583 B (121.1 MiB) | 2,206,821 | 2 |
| test/test_source1.tsv | 175,022,086 B (166.9 MiB) | 1,732,544 | 4 |
| test/test_source2.tsv | 509,456,422 B (485.9 MiB) | 4,887,273 | 4 |
| test/test_source3.tsv | 506,002,772 B (482.6 MiB) | 5,082,316 | 4 |

Total on disk: 2.35 GiB (train 1.24 GiB, test 1.11 GiB). Raw-level integrity check (awk): every line of every file has exactly the expected number of tab-separated fields (4, or 2 for the ground truth), no CR characters, no BOM issues; the files decode as strict UTF-8 with zero errors. A handful of lines contain a literal `"` (train: 4 in S1, 6 in S2; test: 134/349/330) — pandas' default quote handling would mis-parse those, hence `QUOTE_NONE`.

Test-set scale: **1,732,544 Source-1 entities** must each get a row; the candidate pool is 4,887,273 S2 + 5,082,316 S3 = 9,969,589 records.

## 2. SCHEMA

All columns are read as strings; ‘inferred type’ is what the values actually look like. ‘Missing’ = empty / whitespace-only string. **No file contains NaN-style missingness in `entity_id`, `business_name` or `country`; the only truly empty field is `business_address` (S2/S3) and `matched_entity_ids` (singletons in ground truth).** Placeholder strings inside addresses (`NULL`, `<NULL>`, `N/A`) are *not* counted as missing here — see §7/§13.

### train_source1.tsv — 2,206,821 rows × 4 cols

| column | inferred type | example values | missing | missing % | unique |
|---|---|---|---|---|---|
| entity_id | string key `S<k>-<integer 1–9 digits>` | S1-214435868 ; S1-522572257 ; S1-998727651 | 0 | 0.000% | 2,206,821 |
| business_name | free text (Latin + Indic scripts + accents) | Meek Principal ; Family Charities II ; Global Complete Vendome LLC | 0 | 0.000% | 1,539,229 |
| business_address | free text, comma-separated components (Latin + Indic scripts) | Fort Recovery, OH, 2620 Sawmill Road ; 2819 Sanctuary Cove, Katy, TX ; 3106 London Lane, M… | 0 | 0.000% | 2,130,606 |
| country | categorical string | US ; US ; US | 0 | 0.000% | 2 |

### train_source2.tsv — 5,034,616 rows × 4 cols

| column | inferred type | example values | missing | missing % | unique |
|---|---|---|---|---|---|
| entity_id | string key `S<k>-<integer 1–9 digits>` | S2-153809211 ; S2-860628396 ; S2-214575246 | 0 | 0.000% | 5,034,616 |
| business_name | free text (Latin + Indic scripts + accents) | Royal Safe Paceline Inc ; Heritage Éducation Products ; Knight Educational Society Private… | 0 | 0.000% | 4,402,009 |
| business_address | free text, comma-separated components (Latin + Indic scripts) | 414 NORTH STREET, PITTSFIELD CDP, MA ; 1375-D WAGON TRAIL RD, KINGMAN, AZ ; 65 WATERWAY LA… | 168,967 | 3.356% | 4,337,262 |
| country | categorical string | US ; US ; India | 0 | 0.000% | 2 |

### train_source3.tsv — 5,285,603 rows × 4 cols

| column | inferred type | example values | missing | missing % | unique |
|---|---|---|---|---|---|
| entity_id | string key `S<k>-<integer 1–9 digits>` | S3-353598503 ; S3-113330256 ; S3-518854530 | 0 | 0.000% | 5,285,603 |
| business_name | free text (Latin + Indic scripts + accents) | Prairie  Futurecorp Inc ; Associates + Co Care Ltd ; Pediatric Dentistry Group Of  Pittsbu… | 0 | 0.000% | 4,651,609 |
| business_address | free text, comma-separated components (Latin + Indic scripts) | E-1, Industrial Area Ramnagar, Chandauli, UP ; 13219 Eighth Avenue, NULL, Phoenix, Arizona… | 175,916 | 3.328% | 4,632,765 |
| country | categorical string | US ; India ; US | 0 | 0.000% | 2 |

### train_ground_truth.tsv — 2,206,821 rows × 2 cols

| column | inferred type | example values | missing | missing % | unique |
|---|---|---|---|---|---|
| source1_entity_id | string key `S1-<int>` | S1-41288513 ; S1-256559799 ; S1-373624993 | 0 | 0.000% | 2,206,821 |
| matched_entity_ids | comma-separated list of `S2-`/`S3-` keys | S2-669695025,S3-346267754,S3-154538164 ; S2-427072237,S2-194485792,S3-173273014,S3-935… ; … | 123,247 | 5.585% | 2,083,575 |

### test_source1.tsv — 1,732,544 rows × 4 cols

| column | inferred type | example values | missing | missing % | unique |
|---|---|---|---|---|---|
| entity_id | string key `S<k>-<integer 1–9 digits>` | S1-516069816 ; S1-630580376 ; S1-529868590 | 0 | 0.000% | 1,732,544 |
| business_name | free text (Latin + Indic scripts + accents) | Collège Cathare ; High Impex Private Limited ; WFR Multitrade Private Limited | 0 | 0.000% | 1,238,867 |
| business_address | free text, comma-separated components (Latin + Indic scripts) | 20 Rue de Chateaudun, Tourcoing, Hauts-de-Fra… ; Opp. Goyal House, Near Petrol Pump, Ajmer… | 0 | 0.000% | 1,677,483 |
| country | categorical string | France ; India ; India | 0 | 0.000% | 3 |

### test_source2.tsv — 4,887,273 rows × 4 cols

| column | inferred type | example values | missing | missing % | unique |
|---|---|---|---|---|---|
| entity_id | string key `S<k>-<integer 1–9 digits>` | S2-786196841 ; S2-457058927 ; S2-56996144 | 0 | 0.000% | 4,887,273 |
| business_name | free text (Latin + Indic scripts + accents) | Maddox, Thomas & Parker Markets Summit ; Ratnagiri Public School Limited Services ; diraen… | 0 | 0.000% | 4,311,041 |
| business_address | free text, comma-separated components (Latin + Indic scripts) | 4005B 47TH ST, WASIHNGTON, DC ; WEST DES MOINES, 7881-7883 ASPEN DR, IA ; 102 R. JULES GUE… | 129,408 | 2.648% | 4,224,784 |
| country | categorical string | US ; India ; India | 0 | 0.000% | 3 |

### test_source3.tsv — 5,082,316 rows × 4 cols

| column | inferred type | example values | missing | missing % | unique |
|---|---|---|---|---|---|
| entity_id | string key `S<k>-<integer 1–9 digits>` | S3-35552501 ; S3-172385158 ; S3-796255769 | 0 | 0.000% | 5,082,316 |
| business_name | free text (Latin + Indic scripts + accents) | Heritage Heritage Semiconductor Union (Holdin… ; Hari Krishna Tech Traders Private Limited… | 0 | 0.000% | 4,521,929 |
| business_address | free text, comma-separated components (Latin + Indic scripts) | 844 Fernhill Road, Abington Township, Pennsyl… ; H.no 66 C/o Vijay Thakkar, Beside, Meldi … | 136,098 | 2.678% | 4,456,436 |
| country | categorical string | US ; India ; US | 0 | 0.000% | 3 |

## 3. SOURCE 1 ANALYSIS (train_source1.tsv)

- **Total entities:** 2,206,821; **unique entity IDs:** 2,206,821; **duplicate IDs:** 0 (all IDs match `^S1-\d+$`; numeric part 217 … 999,998,822, i.e. random 1–9-digit integers).
- **Unique business names:** 1,539,229 (69.7% of rows); 1,538,804 case-insensitively. **177,793 distinct names occur more than once, covering 845,385 rows (38.3% of S1).** ‘Deduplicated’ therefore means *no exact duplicate records*, not *unique names*: 0 rows share the same (name, address); but the same name is reused by different entities at different addresses (max 253 entities named `Primary Care Group`). Top repeated names: `Primary Care Group` ×253, `Ear Nose & Throat Group` ×251, `Pediatric Group` ×222, `Womens Health Group` ×220, `Physical Therapy Group` ×218, `Pediatric Dental Group` ×216.
- **Unique addresses:** 2,130,606; 40,089 addresses are shared by ≥2 entities (116,304 rows, 5.3%; max 14 entities at one address).
- **Missing name / address / country:** 0 / 0 / 0.
- **Country distribution:** US 1,323,633 (60.0%), India 883,188 (40.0%).
| length (chars) | min | mean | std | p5 | p25 | median | p75 | p95 | p99 | max |
|---|---|---|---|---|---|---|---|---|---|---|
| name | 3 | 24.03 | 7.74 | 12.0 | 18.0 | 24.0 | 30.0 | 37.0 | 42.0 | 105 |
| address | 11 | 52.07 | 25.33 | 27.0 | 33.0 | 41.0 | 70.0 | 103.0 | 124.0 | 256 |

Mean tokens: name 3.55, address 8.03. Country matters for address shape: US addresses average 35 chars / 5.9 tokens (`2620 Sawmill Road, Fort Recovery, OH`), India 78 chars / 11.2 tokens (multi-component, landmarks, sub-localities).

## 4. SOURCE 2 AND SOURCE 3 (same analysis; S1 repeated for comparison)

| metric | S1 | S2 | S3 |
|---|---|---|---|
| total rows | 2,206,821 | 5,034,616 | 5,285,603 |
| unique IDs | 2,206,821 | 5,034,616 | 5,285,603 |
| duplicate IDs | 0 | 0 | 0 |
| unique names | 1,539,229 | 4,402,009 | 4,651,609 |
| unique names (case-insens.) | 1,538,804 | 4,287,470 | 4,579,083 |
| names occurring >1× | 177,793 | 239,779 | 258,276 |
| rows sharing a name | 845,385 | 872,386 | 892,270 |
| unique addresses (incl. '') | 2,130,606 | 4,337,262 | 4,632,765 |
| rows sharing a non-trivial address | 116,304 | 949,860 | 859,813 |
| exact (name,addr) duplicate rows | 0 | 50,969 | 37,283 |
| missing name | 0 | 0 | 0 |
| missing address | 0 | 168,967 | 175,916 |
| missing address % | 0.00% | 3.36% | 3.33% |
| missing country | 0 | 0 | 0 |
| country US | 1,323,633 (60.0%) | 3,016,817 (59.9%) | 3,170,056 (60.0%) |
| country India | 883,188 (40.0%) | 2,017,799 (40.1%) | 2,115,547 (40.0%) |
| name length min | 3 | 2 | 2 |
| name length mean | 24.03 | 25.1 | 25.2 |
| name length p50 | 24.0 | 25.0 | 25.0 |
| name length p95 | 37.0 | 40.0 | 42.0 |
| name length p99 | 42.0 | 48.0 | 50.0 |
| name length max | 105 | 104 | 123 |
| address length min (non-empty) | 11 | 8 | 2 |
| address length mean (non-empty) | 52.07 | 47.83 | 48.32 |
| address length p50 (non-empty) | 41.0 | 37.0 | 42.0 |
| address length p95 (non-empty) | 103.0 | 97.0 | 92.0 |
| address length p99 (non-empty) | 124.0 | 118.0 | 116.0 |
| address length max (non-empty) | 256 | 249 | 240 |
| mean name tokens | 3.55 | 3.5 | 3.53 |
| mean address tokens (incl. empties) | 8.03 | 7.29 | 7.17 |
| names with non-ASCII chars | 0 | 764,608 | 606,737 |
| addresses with non-ASCII chars | 554 | 478,453 | 476,588 |
| names with double space | 0 | 554,392 | 575,616 |

**Reading it:** S2 and S3 are ~2.3–2.4× the size of S1 and are far *less* clean: 3.3–3.4% empty addresses, 11%+ of names contain a double space, ~15% of S2 names and ~11.5% of S3 names contain non-ASCII characters (native Indic script for India records, injected accents such as `Ínnovative` for US records; see §6), and none of S1's names do, and 50 k (S2) / 37 k (S3) rows are exact (name, address) duplicates of another row *in the same source* (§5/§13 — these are always true matches). Top repeated S2/S3 names are generic (`Primary Care` ×320/421, `CC` ×302/387) — they are noise-degraded variants of many different S1 entities, so name alone is not identifying.

Top country notes — S2 by country: US 3,016,817, India 2,017,799; S3: US 3,170,056, India 2,115,547. Missing address by country — S2: US 111,121, India 57,846; S3: US 110,968, India 64,948.

## 5. GROUND TRUTH ANALYSIS (train_ground_truth.tsv)

- Rows: 2,206,821, one per S1 entity; **2,206,821 unique `source1_entity_id`s; all 2,206,821 exist in train_source1 and every train S1 entity appears in the ground truth (0 missing, 0 extra)**. Row order differs from the S1 file order.
- **Malformed rows: 0.** All non-empty match fields match `S[23]-\d+(,S[23]-\d+)*` (no spaces after commas, no stray tokens); 123,247 rows have an empty match list (singletons). Within each list all S2 IDs come before S3 IDs.
- **Every matched ID exists** in the right source: 3,693,619/3,693,619 S2 IDs are in train_source2 and 3,944,746/3,944,746 S3 IDs in train_source3.
- **Duplicate matched IDs: none** — not within a row (0), and not across rows (0 IDs shared between two S1 entities; 7,638,365 unique matched IDs = 7,638,365 pairs). **So each S2/S3 record belongs to at most one S1 entity (the relation is a partition, with ~26% of S2/S3 records left unmatched).** This is a strong structural constraint (§15).

| metric | value |
|---|---|
| Source-1 entities | 2,206,821 |
| with zero matches (singletons) | 123,247 (5.585%) |
| exactly 1 match | 119,157 (5.40%) |
| multiple matches (≥2) | 1,964,417 (89.02%) |
| mean matches / S1 entity | 3.461 (excluding singletons: 3.666) |
| median matches | 3.0 |
| max matches | 11 |
| total positive pairs | 7,638,365 |
| …from S2 | 3,693,619 (mean 1.67/entity, max 5) |
| …from S3 | 3,944,746 (mean 1.79/entity, max 6) |
| entities with matches in BOTH S2 and S3 | 1,776,047 (80.5%; 85.2% of non-singletons) |
| only S2 | 143,029 |
| only S3 | 164,498 |
| S2 records matched / unmatched | 3,693,619 (73.4%) / 1,340,997 |
| S3 records matched / unmatched | 3,944,746 (74.6%) / 1,340,857 |

**Distribution of total matches per S1 entity** (with S2/S3 splits):

| # matches | S1 entities | % |  | # S2 matches | S1 entities | # S3 matches | S1 entities |
|---|---|---|---|---|---|---|---|
| 0 | 123,247 | 5.58% |  | 0 | 287,745 | 0 | 266,276 |
| 1 | 119,157 | 5.40% |  | 1 | 789,108 | 1 | 716,417 |
| 2 | 375,212 | 17.00% |  | 2 | 652,779 | 2 | 668,375 |
| 3 | 530,841 | 24.05% |  | 3 | 333,957 | 3 | 372,443 |
| 4 | 484,115 | 21.94% |  | 4 | 119,078 | 4 | 145,116 |
| 5 | 321,957 | 14.59% |  | 5 | 24,154 | 5 | 35,378 |
| 6 | 164,868 | 7.47% |  |  |  | 6 | 2,816 |
| 7 | 63,968 | 2.90% |  |  |  |  |  |
| 8 | 18,680 | 0.85% |  |  |  |  |  |
| 9 | 4,205 | 0.19% |  |  |  |  |  |
| 10 | 534 | 0.02% |  |  |  |  |  |
| 11 | 37 | 0.00% |  |  |  |  |  |

The distribution is unimodal around 3–4 matches (P(0)=5.6%, P(1)=5.4%, mode 3), roughly ‘each S1 entity is expanded into 0–5 S2 and 0–6 S3 noisy variants’. Only 5.4% of entities have a single match; predicting a small number of matches is *not* the right prior — most entities have 2–5. Matches per entity, by country (train): see §10.


## 6. NAME NOISE ANALYSIS

Method: 500,000 random positive pairs *(sample)*; the S1 name is compared with the matched S2/S3 name. Categories in the first table are **mutually exclusive and priority-ordered** (first rule that fires): identical → case-only → whitespace/punctuation-only → abbreviation/diacritic/`&`-only (`ext` equal) → word-order-only → native-script rewrite → DBA/FKA alias → extra tokens (S1 tokens ⊂ match) → dropped tokens (match ⊂ S1) → otherwise by `token_sort_ratio` band. Because of the priority order, compound noise (typo + suffix change + case change) falls into the lower buckets. The second table lists *independent* pattern flags, which overlap.

| primary relation of matched name to S1 name | all | S2 | S3 | US | India | real examples (S1 → matched) |
|---|---|---|---|---|---|---|
| mixed edits incl. typos, punctuation inside suffix, etc. (token-sort sim ≥ 80) | 16.68% | 16.9% | 16.5% | 17.9% | 14.8% | `First Healthcare LLP`  →  `First Healthcare L.L.P.`<br>`Keystone Drive Inc`  →  `Keystone Dnaige Inc` |
| dropped tokens in match | 15.28% | 15.3% | 15.2% | 16.8% | 13.0% | `Apex Brothers Private Limited`  →  `Apex Brothers Private`<br>`MT Plaza Private Limited`  →  `MT Pláza` |
| whitespace / punctuation only | 11.06% | 10.4% | 11.7% | 12.2% | 9.4% | `Eula's Smart Auto Sales`  →  `Eula's  Smart Auto Sales`<br>`Brightari Twin LLC`  →  `Brightari Twin  LLC` |
| heavy edits incl. domain-style names, honorifics, alias words (50 ≤ sim < 80) | 10.62% | 10.5% | 10.7% | 10.8% | 10.4% | `Natwar Comtech Private Limited`  →  `Dr Natwarcomtech.Com`<br>`Schaeffer Infrastructure Co`  →  `@schaefferin… |
| extra tokens in match (suffix, honorific, duplicated word) | 9.75% | 9.9% | 9.6% | 11.1% | 7.7% | `Tech Services`  →  `Tech Services Co`<br>`Spandan Tree`  →  `Dr Spandan  Tree` |
| native-script rewrite | 7.20% | 9.3% | 5.3% | 0.0% | 18.0% | `Hitech Agro Private Limited`  →  `हाईटेक एग्रो प्राइवेट लिमिटेड`<br>`Pioneer Logistics Limited`  →  `पायोनियर… |
| abbreviation / diacritic / `&`-`and` only | 6.85% | 6.9% | 6.8% | 5.7% | 8.6% | `Alpha & Co Corporation`  →  `Alpha & Co Corp`<br>`Interfaith Fund`  →  `Interfaith Fúnd` |
| case only | 6.09% | 6.3% | 5.9% | 7.6% | 3.8% | `Fowler's Select Bakery`  →  `FOWLER'S SELECT BAKERY`<br>`International Seth Erectors Private Limited`  →  `in… |
| word order only | 6.04% | 6.2% | 5.9% | 6.2% | 5.8% | `New Delhi Transport Pvt. Ltd.`  →  `New Transport Delhi Pvt. Ltd.`<br>`Koch and Hartsfield Kensington LLC`  →… |
| identical | 4.62% | 4.7% | 4.5% | 5.9% | 2.7% | `Kirsten's Consulting`  →  `Kirsten's Consulting`<br>`Kothrud Infra Pvt Ltd`  →  `Kothrud Infra Pvt Ltd` |
| unrelated alias / gibberish / initials (sim < 50) | 4.02% | 3.6% | 4.4% | 3.8% | 4.4% | `J+ Leapfrog LLC`  →  `J+ LLC-Center`<br>`Marketing Jtc Retail Private Limited`  →  `-- #marketingjtc` |
| dba / f/k/a / t/a alias | 1.79% | 0.0% | 3.5% | 2.1% | 1.3% | `XJ Prime Berkley LLC`  →  `Nylaaria formerly XJ Prime Berkley LLC`<br>`American School of Medicine LLC`  →  `… |

**Key observation:** only **4.6%** of true pairs have byte-identical names; 10.75% are equal after lowercasing; 21.9% after `basic`; 34.6% even after `ext`+token-sorting (§8). Roughly two thirds of positives carry *real* string differences that exact/blocked-equality logic cannot resolve.

| pattern (% of names) | S1 side (clean reference) | matched S2 name | matched S3 name |
|---|---|---|---|
| all_upper | 0.0 | 20.62 | 3.01 |
| all_lower | 0.0 | 6.86 | 7.25 |
| double_space | 0.0 | 11.51 | 11.14 |
| leading_junk_punct | 0.14 | 2.49 | 2.38 |
| hash_or_at_prefix | 0.08 | 0.87 | 0.83 |
| honorific_prefix | 0.02 | 3.12 | 3.31 |
| id_suffix | 0.0 | 0.66 | 0.66 |
| has_dba_fka_ta | 0.0 | 0.0 | 3.48 |
| dot_com_domain | 0.0 | 5.22 | 5.14 |
| leet_or_digit_in_word | 0.0 | 2.33 | 2.29 |
| non_ascii_latin | 0.0 | 6.48 | 6.9 |
| native_script | 0.0 | 9.26 | 5.26 |
| duplicate_adjacent_token | 0.12 | 2.16 | 2.11 |
| parenthesized_or_bracketed | 2.06 | 8.02 | 8.23 |
| ampersand | 5.09 | 3.95 | 3.99 |
| word_and | 2.59 | 2.25 | 2.27 |
| plus_sign | 0.15 | 0.56 | 0.66 |
| single_token | 0.69 | 8.22 | 9.88 |
| legal_suffix_present | 67.37 | 50.12 | 51.93 |

**S1 names are the clean canonical form** (no double spaces, no all-caps, no accents, no aliases; ≈67% carry a legal suffix). **All the noise is injected into S2/S3**, with source-specific styles: S2 renders 20.6% of names in ALL CAPS (S3 3%) and has no DBA aliases, while S3 has 3.5% `X dba/fka/t/a <S1 name>` aliases and S3’s US addresses spell out the state name (§7).

### Legal-suffix behaviour

Legal-token set identical to S1’s in 61.01% of pairs; the match **drops** a suffix that S1 has in 22.45%; the match **adds** a suffix S1 lacks in 6.13%. Most frequent transitions (S1 legal tokens → matched-name legal tokens; `-` = none) among the pairs where they differ:

| transition | count (in 500k sample) |
|---|---|
| limited,private -> - | 32,331 |
| llc -> - | 24,576 |
| inc -> - | 18,385 |
| limited,private -> private | 12,927 |
| limited,private -> ltd,private | 10,231 |
| limited -> - | 9,109 |
| limited,private -> limited | 8,451 |
| ltd,pvt -> - | 7,977 |
| - -> inc | 6,313 |
| llp -> - | 4,692 |
| llc -> l.l.c | 4,218 |
| - -> co | 3,841 |
| - -> ltd | 3,830 |
| - -> corporation | 3,753 |

**Which suffix swaps are real?** Measured on 1,000,000 random true pairs: for each abbreviation/full-form couple, the rate at which S1 uses one form and the matched name uses the other *(and not the S1 form)*:

| couple | S1 has abbreviation only → match has full form | S1 has full form only → match has abbreviation | S1 names with abbrev-only / full-only (of 1 M pairs) |
|---|---|---|---|
| `Co` ↔ `Company` | 5.07% | 10.54% | 12,434 / 6,235 |
| `Corp` ↔ `Corporation` | 6.54% | 10.99% | 15,885 / 6,445 |
| `Inc` ↔ `Incorporated` | 4.98% | 0.0% | 107,838 / 2 |
| `Ltd` ↔ `Limited` | 5.8% | 10.67% | 67,267 / 237,246 |
| `Pvt` ↔ `Private` | 0.0% | 0.0% | 54,954 / 196,529 |

**Findings.** (a) `Ltd`/`Limited`, `Corp`/`Corporation`, `Co`/`Company` swap in *both* directions at ≈ 5–11% per occurrence; `Inc → Incorporated` swaps at 5.0% but only in that direction (S1 itself almost never contains `Incorporated`: 2 of 1 M pairs). (b) **`Pvt` ↔ `Private` never flips: 0 swaps in 1,000,000 pairs** (a single case, `Acme Solid Private Limited → Acme Solid Pvt Limited`, was found in a separate 600k-pair scan) — even though the task description lists it. S1 itself uses both forms as separate, stable styles (India S1: `Private Limited` 48.9% of names, `Pvt Ltd` 13.8%) and the matched record keeps S1’s choice (`Private Limited → Private Ltd`, `Private`, `Limited`, dropped; `Pvt Ltd → Pvt Limited`, `Pvt`, dropped, `[PVT]`). (c) In the US `LLC → L.L.C.`, `Inc → Incorporated / Inc.`, `Corp → Corporation`. (d) **A conflicting legal form is almost never noise:** counting pairs where both names carry legal tokens and neither token set contains the other (e.g. `Private` vs `Public`, `LLC` vs `Inc`), true pairs show 0.41% (n=400k) vs **5.4% (n=934) of high-scoring false candidates (p ≥ 0.5) — a ≈13× enrichment** — and 5.9% of true pairs merely differ in some legal token (`Ltd` vs `Limited`). **Dropping/adding a suffix is weak evidence and must not be penalised (dropping is the most common difference, 22%), but a *conflicting* suffix is a strong negative signal.**

### Representative examples by noise type (real S1 → S2/S3 pairs)

| noise type | examples (S1 name → matched name) |
|---|---|
| Pvt ↔ Private | **no true pair found** (0 in 1 M sampled pairs; see the swap-rate table above) — S1 `Pvt Ltd` stays `Pvt Ltd`/… |
| Ltd ↔ Limited | `Al Energy Pvt Ltd`  →  `Al Energy Pvt Limited`<br>`Garden (India) Farms Pvt. Ltd.`  →  `Garden (India) Farms … |
| Corp ↔ Corporation | `Lyra Corp`  →  `Lyra Corporation`<br>`Kannur Outsource Corp`  →  `Kannur Outsource Corporation`<br>`Conant Cu… |
| Inc ↔ Incorporated | `Grand Equity Global Inc`  →  `Grand Equity Global Incorporated`<br>`Jones Blueport Inc.`  →  `Jones Blueport … |
| Co ↔ Company | `Apex Wellness Services Co`  →  `APEX WELLNESS SERVICES COMPANY`<br>`Urology Dynamic Care Associates Co`  →  `… |
| & ↔ and | `Yazzie and Harvey Spark Holdings`  →  `YAZZIE & HARVEY HOLDINGS SERVICES`<br>`Sustain & Associates`  →  `and … |
| LLC ↔ L.L.C. / punctuation in legal form | `Glyphova Algonquin LLC`  →  `Glyphova Algonquin L.L.C.`<br>`Desjardins and Music Calamos LLC`  →  `Desjardins… |
| typos / character edits | `Allied Fund Corp`  →  `Allied Fddi Corp`<br>`Merline Saldivar Morgan LLC`  →  `Merline Saldivar LLC Partners`… |
| digit-for-letter (leet) substitution | `Beacon Fresh Cleaning Service`  →  `Beac0n Fresh Cleaehnign Service`<br>`Sotaor Open LLC`  →  `s0taoropen.com… |
| case/diacritic injection (S1 is ASCII) | `Biex Worldwide PLLC`  →  `PLLC Biex Wórldwide`<br>`Pacific Total Big LLC`  →  `pacific total bíg llc`<br>`Par… |
| domain-style name (`xxx.com`, `www.`) | `Industries Legum Innovations Private Limited`  →  `industriesleguminnovations.com`<br>`Lyra Inc`  →  `Lyra In… |
| honorific / prefix added (Mr, Sri, Shri, Dr, M/s) | `Arkaa Vari Private Limited`  →  `Dr avprivate.com`<br>`Nvn & Associates`  →  `Sri Nvn-& Associates`<br>`Raj F… |
| fake ID suffix `(ID: 12345)` / `#12345` | `Zirain Projects Private Limited`  →  `Zirain Private Projects Limited (ID: 41552)`<br>`Gujarat Construction P… |
| bracketed legal token `[Ltd]`, `(LLC)` | `Kadakkal Holidays Pvt. Ltd.`  →  `Kadakkal Holidays Pvt [Ltd]`<br>`Younkin and Beck`  →  `Younkin and Beck [L… |
| junk prefix (`...`, `--`, `>>`, `[P.C.]`) | `#Zion Electronics, LLC`  →  `#Zion-Electronics, LLC`<br>`Genet Chimera P.C.`  →  `[P.C.] Genet Chmiaear`<br>`… |
| duplicated token | `OON Edutainment Private Limited`  →  `OON Edutainment-Private Limited Limited`<br>`Metropolitan Optics`  →  `… |
| transliteration/translation to native script | `Eastern Logistics Private Limited`  →  `ईस्टर्न लॉजिस्टिक्स प्राइवेट लिमिटेड`<br>`Future Impex Private Limite… |
| DBA / F/K/A / T/A alias (alias name is random, S1 name is inside) | `National Program III`  →  `Veonexio f/k/a National Program III`<br>`Oncology Medicine LLC`  →  `Nexjax dba On… |
| unrelated alias / gibberish or 1–2 letter acronym | `Commercial Electronics International LLC`  →  `Calomira`<br>`Schmidt, Johnson & Garner Ltd`  →  `Déltajax`<br… |

Additional name-noise facts *(sample)*: (i) **word-order changes** are 6.0% as the *only* difference (`Koch and Hartsfield Kensington LLC → Koch LLC Hartsfield Kensington and`) and are also embedded in many of the ‘typo/mixed’ pairs; the legal suffix and `and` are frequently moved into the middle. (ii) **Typos** are keyboard-agnostic random character edits (insert/delete/substitute/transposition; `Keystone Drive → Keystone Dnaige`, `Metropolitan Optics → Metropolitan Otpamcs`) plus digit-for-letter swaps (`El0ise`, `6hazipur`, `lnc` for `Inc`, `5ERVICES`) that appear in ~2.3% of matched names. (iii) **Transliteration**: 9.3% of S2 and 5.3% of S3 matched names are rewritten *entirely in native script* (Devanagari dominant; also Telugu, Kannada, Bengali, Tamil, Gujarati, Malayalam, Gurmukhi, Oriya), i.e. 18% of all India pairs; Latin token overlap with S1 is zero for these, so they can only be matched through the address or through a learned transliteration. They are not literal transliterations of rare words but of the *whole* name, e.g. `Hitech Agro Private Limited → हाईटेक एग्रो प्राइवेट लिमिटेड`. (iv) **DBA/alias patterns** (8,959 in the 500k sample, all in S3): markers `dba`, `formerly`, `f/k/a`, `aka`, `d/b/a`, `t/a`, `trading as`, `fka`, `a/k/a`; in 99.93% of them the true S1 name appears verbatim after the marker (`Veraarcevo+ fka American School of Medicine LLC`), while the prefix is a meaningless invented brand. (v) **Unrelated aliases**: 4.02% of true pairs have `token_sort_ratio`<50 to the S1 name and are not native-script (`Commercial Electronics International LLC → Calomira`, `IT`, `Zetayuma`, `Gildlumxylo`) — for these the name carries **no** information and only the address can match. (vi) **Extra/duplicated tokens** (`Pediatric Pediatric LLC Center`, `Smith LLC Center`) and generic added words (`Center`, `Services`, `Partners`, `Company`) appear on ~10% of pairs; **dropped tokens** on ~15% (`Behavioral Health Medicine of Washington Inc → … Washington`).

Name-similarity distribution of true pairs (`token_sort_ratio` on `ext`-normalised names): p5 = 10.5, p10 = 42.6, p25 = 72.7, median = 89.4, p75 = 100 — i.e. the lower quartile of true pairs looks like a non-match by name.

## 7. ADDRESS NOISE ANALYSIS

Same 500k-pair *(sample)* and the same priority-ordered scheme. Empty matched address is its own category. `partial_overlap` = token-Jaccard ≥ 0.5 after `ext`; `weak_overlap` = 0 < Jaccard < 0.5.

| primary relation of matched address to S1 address | all | S2 | S3 | US | India | real examples (S1 → matched) |
|---|---|---|---|---|---|---|
| partial_overlap(jaccard>=0.5) | 40.91% | 26.4% | 54.6% | 45.5% | 34.0% | `Memphis, 1473 Winfield Avenue, TN`  →  `1473 Winfield Avenue, Memphis, Tennessee`<br>`No.3, Arcot Road, I Flo… |
| weak_overlap(0<jaccard<0.5) | 14.67% | 4.9% | 23.8% | 16.5% | 11.9% | `Central Point, 881 Westrop Drive, Fl 1, OR`  →  `881 Westrop Dr, PMB 3918, Central Point, Oregon`<br>`2033, S… |
| native_script_component | 9.05% | 9.4% | 8.8% | 0.0% | 22.6% | `19-2-226/1/E Bahadurpura, Bahadurpura, Hyderabad, Telangana`  →  `9-2-226/1/E Bahadurpura, Bahadurpura, Hyder… |
| match_missing_components(subset of S1) | 8.53% | 17.2% | 0.3% | 7.6% | 10.0% | `14210 Shoreview Drive, Medical Lake, WA`  →  `SHOREVIEW DR, MEDICAL LAKE, WA`<br>`510 Harvard Lane, Hoffman E… |
| case/whitespace/punct_only | 6.11% | 12.6% | 0.0% | 6.5% | 5.5% | `780 F M Watts Road, Whiteville, NC`  →  `780 F M WATTS ROAD, WHITEVILLE, NC`<br>`Tf 1301 Pehel Lakeview, At P… |
| match_has_extra_components(PO Box/PMB/county...) | 5.16% | 8.5% | 2.0% | 3.9% | 7.1% | `404 Bldg-2D, Mumbai, Powai Vihar Bldg No.2 Chs Ltd, A S Marg…`  →  `404/8 BLDG-2D, POWAI VIHAR BLDG NO.2 CHS … |
| component_reorder_only | 5.04% | 8.4% | 1.9% | 6.5% | 2.8% | `1701 Flemming Road, OH, Middletown`  →  `1701 FLEMMING ROAD, MIDDLETOWN, OH`<br>`House No.21, Block 4, Tyagi … |
| empty_address | 4.43% | 4.5% | 4.3% | 4.8% | 4.0% | `414 Avenshire Court, Leland, NC`  →  ``<br>`107 1st Avenue, Tripoli, IA`  →  `` |
| abbrev/placeholder_only(Rd/Road, NULL dropped) | 3.88% | 8.0% | 0.0% | 6.4% | 0.1% | `4028 Langhorne Avenue, Charlotte, NC`  →  `4028 LANGHORNE AVE, null, CHARLOTTE, NC`<br>`1532 Main Street, Ott… |
| identical | 2.22% | 0.0% | 4.3% | 2.3% | 2.1% | `102 A Swinhoe Lane Landmark Kalyan Sangha Club, Kolkata, How…`  →  `102 A Swinhoe Lane Landmark Kalyan Sangha… |
| no_token_overlap | 0.00% | 0.0% | 0.0% | 0.0% | 0.0% | `Bangalore South, Karnataka, Turquoise By Src, 2Nd Floor, Bil…`  →  `91, KA`<br>`335Ff, Sector-32, Pi Gulmohar… |

**Key observations.** (a) Only 2.2% of matched addresses are byte-identical (4.3% of S3 pairs; **0% of S2 pairs**, whose addresses are always re-cased/re-formatted) and 8.3% are equal after `basic`. (b) **The two sources apply different transformations** (state formats measured on 400k-row samples). *Both* S2 and S3 abbreviate street types (~45–49% of US addresses), insert `NULL`/`N/A` fillers (~3–4%), zero-pad house numbers (~4.5–4.9%), add `PO BOX`/`PMB` (~1.6–1.7% of US) and `#`/unit tokens. **S2** additionally is ALL-CAPS (93% of US S2 addresses; 25% of India), keeps the US state as a 2-letter code (like S1) and the India state as a full Latin name (77%) or native script (21%), shuffles component order (8.4% reorder-only vs 1.9% in S3) and drops components (17% ‘subset of S1’ vs 0.3% in S3). **S3** is mixed-case, **spells out the US state** (95% of US S3 addresses; `Texas` vs S1 `TX`), writes the **India state as a 2-letter code** (~70% of India S3 addresses: `GJ`, `UP`, `TN`) or in native script (~20%), and substitutes/adds districts and cities (54.6% partial overlap, 23.8% weak overlap vs 26.4%/4.9% in S2). A state/city normaliser therefore needs code↔name↔native-script maps (learnable from the training pairs). (c) Address token-Jaccard between true pairs (non-empty): p5 0.30, p25 0.56, median 0.71, p75 0.87 — high but far from 1. (d) A true pair with **zero** shared address token is essentially nonexistent (0.005% of non-empty pairs) — *every non-empty matched address still shares at least one token with the S1 address*, which is why address tokens are such a strong blocking signal (§11).

| pattern (% of non-empty addresses) | S1 US | S2 US | S3 US | S1 India | S2 India | S3 India |
|---|---|---|---|---|---|---|
| all_upper | 0.0 | 93.28 | 0.0 | 0.0 | 24.53 | 0.05 |
| abbr_Rd/St/Ave/Dr/Ln/Ct/Cir/Blvd/Hwy | 3.2 | 48.67 | 44.96 | 3.25 | 2.91 | 2.38 |
| full_Road/Street/Avenue/Drive/Lane/Court | 87.15 | 39.7 | 42.68 | 31.05 | 27.58 | 21.69 |
| US_full_state_name_at_end | 0.04 | 0.0 | 53.99 | 0.0 | 0.0 | 0.0 |
| NULL/N/A/<NULL>_placeholder | 0.0 | 3.91 | 3.74 | 0.0 | 3.02 | 2.67 |
| PO_BOX | 0.0 | 1.68 | 1.58 | 0.0 | 0.0 | 0.0 |
| PMB | 0.0 | 1.74 | 1.57 | 0.0 | 0.0 | 0.0 |
| unit_or_# | 13.5 | 5.23 | 15.09 | 3.08 | 12.48 | 11.32 |
| leading_zero_number | 0.02 | 4.91 | 4.47 | 0.46 | 2.69 | 2.48 |
| leading_hash | 0.0 | 4.53 | 4.54 | 0.87 | 5.87 | 5.22 |
| fraction_slash_1/2 | 0.12 | 2.09 | 2.04 | 0.4 | 0.38 | 0.39 |
| suffix_letter_after_number(3707-D) | 0.06 | 3.46 | 3.22 | 2.24 | 1.5 | 1.45 |
| landmark(near/opp/behind/beside/next to) | 0.01 | 0.01 | 0.01 | 11.49 | 10.15 | 7.65 |
| native_script | 0.0 | 0.0 | 0.0 | 0.0 | 24.36 | 22.74 |
| double_space | 0.08 | 2.72 | 0.15 | 0.0 | 1.16 | 0.0 |

Reading the table: **Rd/Road, St/Street** — S1 US spells out 87% of street types (`Road`, `Street`), S2/S3 abbreviate ~45–49% (`RD`, `ST`, `AVE`) → an abbreviation dictionary is worthwhile. **Landmarks** (`Near/Opp/Behind/Beside`) are an India phenomenon (11.5% of S1-India addresses, 0.01% in US) and appear intact in S2 (`NEAR MDSD GIRLS COLLEGE`) but are *sometimes dropped* in S3 (7.7% of S3-India addresses contain a landmark keyword vs 11.5% in S1; e.g. `Plt No. 6, Chandrapur, Gadchandur, MH`). **Native script** appears in 22–24% of India S2/S3 addresses, almost always only for the *state* (`महाराष्ट्र`, `తెలంగాణ`, `ગુજરાત`, `தமிழ்நாடு`) and sometimes for the whole tail; US has none. **Municipal numbering** (`6-3-907/912/C`, `H.no 910 A 3503`, `1-1-770/A`, `C-46`) is heavily used in India and is preserved verbatim in most variants (the first number is unchanged in 79% of pairs, see below), making it a highly discriminative substring, but S3/S2 sometimes prepend/alter it (`No 85 67`, `HN A-834`, `#449`, `DORO NO 0689`).

### Missing / altered components — measured *(sample: 400k US pairs per source)*

| US pairs | S2 match | S3 match |
|---|---|---|
| S1 address has a state | 100% | 100% |
| **state missing** in match (when S1 has it) | 0.0% | 15.19% |
| S1 city found verbatim in match | 82.59% | 83.76% |
| **city missing or misspelled** (fuzzy≥85 fails) | 11.74% | 10.98% |
| city present but misspelled (`BALCH SPRINS`, `Clolierville`) | 5.66% | 5.26% |
| street number present in match (S1: 100%) | 91.49% | 92.78% |
| ZIP code present (either side) | 0% | 0% |

**PIN / postal codes are absent from the data.** No source, in either train or test, contains ZIP codes (0.000% of US addresses end in `ST 12345`), Indian PIN codes (6-digit tokens appear in only 0.3–0.8% of addresses and are plot/survey numbers, not PINs), or French code postaux (`59000 Lille` pattern ≈ 0%). *There is no postal-code blocking key.* The geographic signals available are state, city, district/locality strings and the house/plot number.

**House / plot number** (first numeric token; both addresses have one in 87.17% of pairs): equal in 79.09% of those pairs, 83.19% after stripping leading zeros (US 87.65%, India 76.4%). So the number is wrong or different in ~17% of true pairs: the noise process *mutates* numbers as well (`3979 → 979`, `229 → 82`, `128 → 28`, `13510 → 013510`, `73 → 73 1/2`, `3707 → 3707-D`). A house-number blocking key would lose ~1 in 6 true pairs.

| address noise type | real examples (S1 → matched) |
|---|---|
| Rd<->Road / St<->Street (abbrev vs full) | `149 Grant Street, Fall River, MA`  →  `149 GRANT ST, FALL RIVER, MA`<br>`Sioux Center, 1330 Main Avenue, IA` … |
| missing city/state | `23 Spruce Road, Frenchboro, ME`  →  `ME, FRENCHBORO, SPRUCE ROAD`<br>`6900 191st Place, Arlington, WA`  →  `1… |
| missing state (US) - only street+city | `36 Market Street, New York, NY`  →  `36 Market St, New York`<br>`938 8 Avenue, New York, NY`  →  `938 8 Ave, … |
| reordered components | `Kansas City, 7029 Verde Drive, KS`  →  `7029 VERDE DRIVE, KANSAS CITY, KS`<br>`Sioux Falls, 2000 Kinderhook A… |
| landmark (India) | `Maharashtra, Chandrapur, Survey No. 8/1, Asset Sr No. 2108 W…`  →  `Plt No. 6, Chandrapur, Gadchandur, MH`<br… |
| PO Box/PMB inserted in match | `52 Bennett Street, Taunton, MA`  →  `52 BENNETT ST, PMB 3364, TAUNTON, MA`<br>`2129 Lansing Avenue, Tulsa, OK… |
| NULL/N/A placeholder | `Mumbai, Lamington Road, Maharashtra, 1410/Iii Navjivan Comme…`  →  `1410/III NAVJIVAN COMMERCIAL PREMISES SOC… |
| leading-zero house number | `4494 Redcedar Road, Mcleansville, NC`  →  `004494 Redcedar Rd, Mcleansville, North Carolina`<br>`7427 Seneca … |
| house number differs (first number) | `13510 Driftwood Drive, Victorville, CA`  →  `013510 DRIFTWOOD DRIVE, VICTORILLE, CA`<br>`812 Winged Foot Lane… |
| city misspelled / substituted | `358 Lyon Street, Elmira, NY`  →  `358 LYON ST, ELMIA, NY`<br>`1413 Prescott Place, Fl 1, Chandler, AZ`  →  `1… |
| native-script state/city (India) | `Plot No. 109D, Mahendra Ind.Estate, Iiird Flr Road No.29, Si…`  →  `H.NO 109D, MAHENDRA IND.ESTATE, IIIRD FLR… |
| US full state vs abbreviation | `113 Brittany Drive, Royse City, TX`  →  `113 Brittany Dr, Royse City, Texas`<br>`104 Pats Court, Weatherford,… |
| India state code vs full (GJ/Gujarat) | `C/O G Narayanan, No.9, Valmigi Street Periyar Nagar, Pallika…`  →  `C/o G Narayanan, No.9, Valmigi Street Per… |
| India transliteration variants (spelling) | `No.735, 13Th Cross, 7Th Block Jayanagar, Bangalore South, Ba…`  →  `No.735., 13Th Cross, 7Th Block Jayanagar,… |
| no token overlap | `T5, H.No: 702 Gera Skyvillas Chs, S No-64 (1 To 6) Off Pune-…`  →  `T05, Poona, Wagholi P, MH`<br>`Rajkot, 13… |

Transliteration variants of place names occur in India (`Ahmadabad`/`Ahmedabad`, `Kolkta`, `Poona`/`Pune`, `Madras`/`Chennai`, `Bangalore`/`Bengaluru`, `Eranakulam`/`Ernakulam`, `Keralam`/`Kerala`, `Vishakhapatnam`/`Visakhapatnam`, `Gautam Budh Nagar`/`Gautam Buddha Nagar`), and *city substitution* to a neighbouring district (`Rajkot → Porbandar`, `New Delhi → Gautam Buddha Nagar`, `Phoenix → NORTHEAST` (AZ), `Toledo → ‘Tledo’`) means even the city is not reliable. Long Indian addresses (p95 = 103 chars, up to 268) carry repeated sub-localities that are shuffled or dropped.

## 8. EXACT MATCH ANALYSIS (all 7,638,365 training positives)

Computed on **every** true pair (not a sample), with the normalisations defined at the top. ‘Address’ figures require a non-empty matched address (4.4% of positives have an empty address and can never match on it).

| normalisation | name equal | address equal | BOTH equal | name OR address equal | NEITHER equal |
|---|---|---|---|---|---|
| raw (byte-identical) | 4.64% | 2.23% | — | — | — |
| lowercase only | 10.75% | — | — | — | — |
| **basic** (lower + punct→space + whitespace) | 21.85% | 8.27% | 1.31% | 28.8% | **71.2%** |
| **ext** (basic + `&`→and + diacritic fold + abbreviations) | 28.48% | 12.13% | 2.91% | 37.7% | **62.3%** |
| **sorted** (ext + token-sorted → order-invariant) | 34.57% | 17.16% | 4.97% | 46.76% | **53.24%** |

Name-only equal (basic): 20.54%; address-only equal (basic): 6.96%. Address is empty for 4.41% of positives; the matched record is in the same country for 100.0%.

**Interpretation.** Even with the most generous deterministic normalisation, **53% of true matches share neither an identical name nor an identical address** (71% under plain `basic`), and only 5% share both. Exact-key joins can therefore recover at best ~47% of matches (the rest need fuzzy comparison) — while by construction they are the safest matches. Breakdown of the `basic` numbers: by source — name 21.4% (S2) / 22.2% (S3), address 12.5% (S2) / **4.3% (S3)** (S3 formats addresses more differently: full state names, dropped components); under `sorted`, S2 address equality reaches 28.9% but S3 only 6.2%. By country (S1 side) — name basic-equal India 15.9% / US 25.8%; address `sorted`-equal India 10.5% / US 21.6%. India is harder for exact logic because of native-script names, longer multi-component addresses and heavier number/locality perturbation.

**Exact matches are *not* automatically unambiguous.** Because 38% of S1 rows share their name with another S1 entity (§3), an exact-name hit has to be disambiguated by address: in the hard-negative test of §9, an S1 name paired with the match of a *different* S1 entity that has the identical name gets `token_sort_ratio` ≈ 0.72 on average and the name features cannot separate them (AUC ≈ 0.55).


## 9. NAME (AND ADDRESS) SIMILARITY ANALYSIS — positives vs negatives

**Design** *(samples; features are computed on `ext`-normalised strings)*. Positives: 150,000 random true pairs. Negatives (four kinds, because the choice of negative decides how ‘separable’ things look):

| negative type | n | how it is built |
|---|---|---|
| `random_same_country` | 150,000 | random S1 entity × random S2/S3 record of the same country (includes unmatched records). *This is the easy cas… |
| `hard_same_block` | 146,919 | S1 × a non-matching S2/S3 record that shares country + first 4 chars of the first name token (a typical name-b… |
| `hard_same_name_diff_entity` | 75,000 | S1 entity *a* × a true match of a **different** S1 entity *b* that has the same normalised name as *a* (name-t… |
| `hard_same_addr_diff_entity` | 75,000 | same as above but *a* and *b* share the same normalised address |

No sampled negative is a true pair (checked: 0). Features: character similarity (`fuzz.ratio`, `partial_ratio`), token-order-insensitive (`token_sort_ratio`, `token_set_ratio`), edit-distance (normalised Levenshtein), Jaro-Winkler, token Jaccard / overlap-coefficient, char-3-gram Jaccard, and TF-IDF cosine (word-level and char-3-gram; vectorisers fitted on a 600k-string sample of S1+S2+S3). Address features are the same plus a house-number-equality flag, and are `NaN` when the matched address is empty.

| feature (mean) | positive | random neg | hard: same block | hard: same name | hard: same addr |
|---|---|---|---|---|---|
| n_ratio | 0.792 | 0.347 | 0.536 | 0.708 | 0.343 |
| n_tsort | 0.804 | 0.342 | 0.501 | 0.717 | 0.338 |
| n_tset | 0.874 | 0.359 | 0.572 | 0.782 | 0.354 |
| n_jw | 0.881 | 0.552 | 0.745 | 0.822 | 0.548 |
| n_lev | 0.711 | 0.223 | 0.414 | 0.636 | 0.218 |
| n_jacc | 0.660 | 0.046 | 0.202 | 0.576 | 0.043 |
| n_c3jacc | 0.672 | 0.047 | 0.173 | 0.589 | 0.045 |
| n_tfidf_word | 0.767 | 0.017 | 0.325 | 0.669 | 0.017 |
| n_tfidf_char | 0.789 | 0.038 | 0.367 | 0.687 | 0.037 |
| a_tsort | 0.862 | 0.365 | 0.370 | 0.366 | 0.865 |
| a_tset | 0.934 | 0.372 | 0.379 | 0.375 | 0.934 |
| a_jacc | 0.694 | 0.020 | 0.025 | 0.023 | 0.693 |
| a_tfidf_word | 0.824 | 0.012 | 0.018 | 0.016 | 0.825 |
| a_tfidf_char | 0.837 | 0.034 | 0.042 | 0.039 | 0.837 |
| a_housenum_eq | 0.760 | 0.004 | 0.005 | 0.005 | 0.780 |

Positives: name `token_sort_ratio` mean 0.80 / median 0.89 (p25 0.73, p5 0.11) vs 0.34 for random negatives; address `token_set_ratio` mean 0.93 vs 0.37. The **positive distribution is heavily skewed with a fat lower tail**: ~66% of positives score ≥0.8 on name, but ~5% are near zero (p5 = 0.11) and ~10% are below 0.43 (native script / unrelated alias / heavy edits) — the p5 value is essentially an ‘unrelated’ name.

**Single-feature ROC-AUC, positive vs each negative type** (0.5 = no separation; the ‘same addr’ column is meaningless for address features by construction):

| feature | vs random | vs same-block | vs same-name/diff-entity | vs same-addr/diff-entity | vs all negatives |
|---|---|---|---|---|---|
| n_ratio | 0.911 | 0.854 | 0.556 | 0.911 | 0.833 |
| n_tsort | 0.910 | 0.863 | 0.559 | 0.910 | 0.836 |
| n_tset | 0.913 | 0.863 | 0.555 | 0.913 | 0.837 |
| n_jw | 0.916 | 0.829 | 0.556 | 0.916 | 0.827 |
| n_lev | 0.902 | 0.821 | 0.554 | 0.903 | 0.817 |
| n_jacc | 0.912 | 0.849 | 0.556 | 0.913 | 0.832 |
| n_c3jacc | 0.937 | 0.889 | 0.557 | 0.938 | 0.858 |
| n_tfidf_word | 0.912 | 0.843 | 0.558 | 0.913 | 0.830 |
| n_tfidf_char | 0.940 | 0.880 | 0.560 | 0.941 | 0.857 |
| a_tsort | 0.945 | 0.943 | 0.945 | 0.495 | 0.869 |
| a_tset | 0.956 | 0.955 | 0.956 | 0.501 | 0.879 |
| a_jacc | 0.956 | 0.955 | 0.956 | 0.502 | 0.879 |
| a_tfidf_word | 0.956 | 0.955 | 0.956 | 0.499 | 0.879 |
| a_tfidf_char | 0.956 | 0.955 | 0.956 | 0.500 | 0.879 |
| a_housenum_eq | 0.843 | 0.844 | 0.844 | 0.491 | 0.784 |

The same restricted to pairs whose matched name is **not** native script (Latin-script names only):

| feature | vs random | vs same-block | vs same-name/diff-entity | vs same-addr/diff-entity |
|---|---|---|---|---|
| n_tsort | 0.970 | 0.927 | 0.503 | 0.971 |
| n_tset | 0.973 | 0.926 | 0.498 | 0.974 |
| n_jw | 0.976 | 0.891 | 0.500 | 0.977 |
| n_lev | 0.958 | 0.880 | 0.498 | 0.960 |
| n_jacc | 0.948 | 0.901 | 0.506 | 0.949 |
| n_c3jacc | 0.982 | 0.952 | 0.502 | 0.982 |
| n_tfidf_word | 0.948 | 0.895 | 0.508 | 0.949 |
| n_tfidf_char | 0.987 | 0.945 | 0.506 | 0.986 |

**What this says**

1. **Names separate well against random and blocked negatives** (AUC 0.90–0.94 over all positives; **0.95–0.99 when the matched name is Latin-script**, with char-3-gram / TF-IDF-char / partial-ratio best and token-Jaccard / Levenshtein worst), but **only ~0.83 against all negatives overall** because native-script and alias names (~11% of positives) score like negatives. Character-n-gram measures beat token measures (0.987 vs 0.948 for TF-IDF char vs word against random negatives, Latin names) — the typos/leet/accents break exact tokens but not most trigrams.
2. **Names cannot separate same-name twins** (AUC 0.50–0.56 for every name feature): when two S1 entities share a name, only the address can tell whose record it is. 38% of S1 rows are in such groups.
3. **Addresses separate at AUC 0.94–0.96 against random/blocked negatives** (TF-IDF, token-set and Jaccard best; Levenshtein weakest 0.90 because of reordering) and are the *only* signal against name-twins (0.945–0.985) — but fail on the ~4.4% empty addresses and are blind to same-address twins (0.50), where the name decides. **The two fields are complementary, not redundant.**
4. Coverage of positives by simple thresholds: name `token_sort_ratio`≥0.8 for 66.4%; address ≥0.8 for 74.2%; **either** for 91.3%; neither for 8.75%; name<0.5 but address ≥0.7 for 9.0% (rescued only by address); name≥0.8 but address <0.5/empty for 5.3% (rescued only by name); **name<0.5 and address weak/empty for only ~1.0%** (near-hopeless pairs).

**Best single-feature F0.5 at an optimal threshold** (balanced positives/negatives sample, so absolute precision is *optimistic* vs real prevalence): name `token_set_ratio` reaches F0.5 = 0.96 vs random negatives but only 0.73 vs same-name twins; address TF-IDF reaches 0.987 vs random and 0.982 vs same-block, but 0.714 vs same-address twins.

**Combined separability (diagnostic only — a small gradient-boosting model on the ~20 features above, trained on 70% of these pairs, evaluated on 30%; this is a probe of separability, not a proposed model):**

| evaluated against | AUC | precision @p≥0.5 / 0.8 / 0.9 / 0.95 | recall @ same thresholds |
|---|---|---|---|
| random_same_country | 1.0 | 1.000 / 1.000 / 1.000 / 1.000 | 0.984 / 0.968 / 0.915 / 0.870 |
| hard_same_block(country+name[:4]) | 0.9992 | 0.993 / 0.996 / 0.997 / 0.998 | 0.984 / 0.968 / 0.915 / 0.870 |
| hard_same_name_diff_entity | 0.997 | 0.978 / 0.981 / 0.998 / 0.999 | 0.984 / 0.968 / 0.915 / 0.870 |
| hard_same_addr_diff_entity | 0.9947 | 0.966 / 0.984 / 0.990 / 0.997 | 0.984 / 0.968 / 0.915 / 0.870 |

Combined AUC 0.998 over all negatives. **Recall at p≥0.9 on positives: S2 0.916, S3 0.914; US 0.93, India 0.892; native-script names 0.736 vs Latin 0.928; empty matched address 0.062 (!) vs non-empty 0.956.** So the pairs are *highly separable when both fields are present* (>95% recall at very high precision); the unrecoverable mass is (i) empty-address positives (4.4%; recall ~6%, name only) and (ii) native-script/alias names. Caveat: these negatives are sampled, not the real candidate stream; §11–12 repeat the exercise on realistic retrieved candidates, where separation is noticeably harder (24% of singletons still receive a false candidate with p≥0.5).

## 10. COUNTRY ANALYSIS

| file | rows | US | India | France |
|---|---|---|---|---|
| train S1 | 2,206,821 | 1,323,633 (60.0%) | 883,188 (40.0%) | 0 (0.0%) |
| train S2 | 5,034,616 | 3,016,817 (59.9%) | 2,017,799 (40.1%) | 0 (0.0%) |
| train S3 | 5,285,603 | 3,170,056 (60.0%) | 2,115,547 (40.0%) | 0 (0.0%) |
| test S1 | 1,732,544 | 663,106 (38.3%) | 809,986 (46.8%) | 259,452 (15.0%) |
| test S2 | 4,887,273 | 1,871,330 (38.3%) | 2,312,565 (47.3%) | 703,378 (14.4%) |
| test S3 | 5,082,316 | 1,945,701 (38.3%) | 2,405,000 (47.3%) | 731,615 (14.4%) |

- **Countries in training data: exactly two, `US` and `India`.** The `country` field is fully populated (0 missing) with clean, canonical labels in every file — no case/spacing variants (`us`, `USA`, `IN`, …); `repr()` of the unique values shows only `'US'`, `'India'` (train) and additionally `'France'` (test S1/S2/S3). **France exists only in test** (259,452 S1 / 703,378 S2 / 731,615 S3 = 15.0% / 14.4% / 14.4% of test records) and has no labelled example. Treat `country` as an open string key.
- **Country distribution by source is stable across S1/S2/S3** in train (US ≈ 60.0%, India ≈ 40.0%): S1 59.98% US, S2 59.92%, S3 59.98%; the test US/India split is 38.3/46.8 with France 15.0% for S1, 38.3/47.3/14.4 for S2 and 38.3/47.3/14.4 for S3 — i.e. **test is not a shrunken train**: the US share falls from 60% to 38%, India rises 40%→47%, and France adds 14–15%.
| S1 country | S1 entities | positive pairs to S2 | to S3 | mean matches / entity | singleton rate | max |
|---|---|---|---|---|---|---|
| India | 883,188 | 1,480,545 | 1,579,298 | 3.465 | 5.59% | 11 |
| US | 1,323,633 | 2,213,074 | 2,365,448 | 3.459 | 5.58% | 11 |

- **Matched pairs by country:** every one of the 7,638,365 training positives has `country(S1) == country(S2/S3)` (**100.00%**; 0 pairs cross countries) — there are **no examples where the country differs**. Same-country blocking is therefore lossless in train and cuts the candidate pool by ~40–60% (US pool 6.19 M, India 4.13 M in train; test pools US 3.82 M, India 4.72 M, France 1.43 M).
- Match density is the same in both training countries: 3.46 matches per entity, 5.6% singletons, and the fraction of S2/S3 records that are matched is 73.4%/74.6% in both US and India (S2 US 73.36%, India 73.37%; S3 US 74.62%, India 74.65%). The generator is country-symmetric; **only noise style differs** (India: native-script names/addresses, longer addresses with landmarks; US: ZIP-less street addresses, `PO Box`/`PMB`).
- **France (test only) — what can be learned without labels** (§14 has more): Latin script with heavy accent use (15.7% of S1 names and 73k S1 addresses contain accented letters), French legal forms (`SARL` 36%, `SAS` 25%, `EURL`, `SASU`, `SCI`, `SA`, `SNC`; `S.A.S`/`S.A.R.L` dotted variants only in S2/S3), association-type words (`Club`, `Amicale`, `Comité`, `École`, `Maison`, `Établissements`/`Ets`) and short template names (mean 19.4 chars vs 24 in train). Addresses use `Rue/Avenue/Allée/Boulevard/Impasse` with `R.`, `Av`, `Bd`, `Pl.` abbreviations in S2/S3 and roughly 15 cities account for most addresses (Bordeaux, Nantes, Lille, Tourcoing, Dunkerque, Roubaix, Calais, Saint-Nazaire, Pessac, La Teste-de-Buch, Mérignac, Lège-Cap-Ferret, Pornic, Saint-Herblain…) with region names (Hauts-de-France, Nouvelle-Aquitaine, Pays de la Loire) — **so city/region tokens carry almost no discriminative information in France; street name + number must do the work.**

**Consequence for design:** never one-hot or filter on `{US, India}`; block within `country` as an opaque string; and make every learned component (abbreviation dictionaries, IDF weights, thresholds) either country-conditional with a *global fallback* or unsupervised-fit on test text, because a model trained only on US/India will have seen none of France’s vocabulary (53% of French S1 name tokens and 37% of address tokens never occur in any training file).


## 11. CANDIDATE / BLOCKING INSIGHTS

### 11.1 Size of the problem

|  | all pairs S1×(S2∪S3) | same-country pairs | per S1 entity (same-country pool) |
|---|---|---|---|
| train | 2.277e+13 | 1.184e+13 | US 6,186,873 · India 4,133,346 records |
| test | 1.727e+13 | 6.725e+12 | India 4,717,565 · US 3,817,031 · France 1,434,993 records |

Brute force is impossible (10¹³ pairs). The numbers below evaluate each blocking family on **training positives (recall) and on the real S2∪S3 pool of the same country (candidate counts)**. Test pools are the same order of magnitude (US 38% smaller, India 14% larger than train) so candidate counts transfer approximately. Recall = share of the true (S1, S2/S3) pairs that would survive the blocker; ‘cands’ = number of S2+S3 records returned per S1 entity.

### 11.2 Same-country blocking

Recall **100.00%** (7,638,365/7,638,365 training positives are same-country); pool per S1 drops from 10.3 M to 6.2 M (US) / 4.1 M (India) in train. Necessary and free, but leaves 10⁶ candidates per entity — it is a *filter*, not a blocker. Because France has no training data, the safe implementation is `country` string equality (open set).

### 11.3 Single-key blocking (exact key equality; recall from 200k sampled positives per country, candidates exact over the full pool)

**US** (S1 entities 1,323,633; pool 6,186,873)

| key (within country) | recall | mean cands / S1 | median | p99 | S1 with 0 cands |
|---|---|---|---|---|---|
| name_first_token | 81.3% | 6,477 | 1,869 | 59,168 | 0.01% |
| name_first3chars(nospace) | 86.7% | 13,792 | 10,767 | 62,963 | 0.0% |
| name_first4chars(nospace) | 85.9% | 8,321 | 3,769 | 62,109 | 0.04% |
| name_first6chars(nospace) | 82.4% | 5,452 | 476 | 62,029 | 0.19% |
| name_first_token[:4] | 86.6% | 8,755 | 4,445 | 62,104 | 0.0% |
| name_min_token_alpha | 72.4% | 73,971 | 9,009 | 482,497 | 0.03% |
| name_sorted_first2tokens | 55.5% | 892 | 51 | 17,914 | 1.87% |
| addr_first_number | 77.0% | 2,752 | 953 | 15,833 | 0.15% |
| state | 93.3% | 272,000 | 255,796 | 598,958 | 1.74% |
| state+name_first3 | 80.3% | 611 | 320 | 3,777 | 1.83% |
| state+addr_first_number | 75.3% | 230 | 40 | 3,277 | 2.43% |
| addr_first_number+name_first3 | 65.9% | 8 | 4 | 71 | 6.38% |
| addr_first_number+name_first_token[:4] | 65.8% | 6 | 3 | 52 | 7.78% |

**India** (S1 entities 883,188; pool 4,133,346)

| key (within country) | recall | mean cands / S1 | median | p99 | S1 with 0 cands |
|---|---|---|---|---|---|
| name_first_token | 60.1% | 3,812 | 727 | 27,971 | 0.01% |
| name_first3chars(nospace) | 63.9% | 9,807 | 6,881 | 80,218 | 0.0% |
| name_first4chars(nospace) | 62.4% | 4,854 | 1,568 | 29,940 | 0.11% |
| name_first6chars(nospace) | 58.8% | 1,964 | 190 | 19,623 | 0.47% |
| name_first_token[:4] | 63.4% | 5,482 | 4,690 | 29,925 | 0.01% |
| name_min_token_alpha | 59.2% | 58,122 | 9,229 | 313,073 | 0.03% |
| name_sorted_first2tokens | 45.4% | 8,211 | 117 | 103,298 | 1.8% |
| addr_first_number | 65.8% | 39,371 | 11,061 | 184,020 | 8.95% |
| addr_first_number+name_first3 | 41.4% | 95 | 11 | 1,087 | 13.13% |
| addr_first_number+name_first_token[:4] | 41.1% | 54 | 6 | 655 | 14.44% |

**Take-aways.** (1) *First characters/tokens of the normalised name* are cheap and reach 81–87% recall in the US but return **≈5,500–14,000 candidates per S1** (the name vocabulary is small and templated: `Primary`, `Global`, …); in India recall is only 59–64% because ~18% of India positives have native-script names whose first characters cannot match a Latin S1 name. Longer prefixes trade recall for size only slowly (US first-6-chars: 82% recall, still 5,452 mean/476 median). (2) `min-token` / `longest-token` keys are hopeless (skew: up to 843k candidates). (3) **House number alone**: 77% (US) / 66% (India) recall — bounded by number mutation (~17%) and the 4.4% empty addresses — with 2,750 (US) / 39,000 (India) candidates. (4) The only keys that give tiny candidate lists are **conjunctions** (number + name prefix: 6–8 candidates in the US) and they lose a third to 60% of the positives (recall 66% US, 41% India). (5) `state` reaches 93% recall in the US (it loses the 4.7% empty addresses and the ~15% of S3 addresses that drop the state) but returns 272k candidates; state combined with a name prefix is 80% / 611 candidates. **No single key works; the signal is spread across name tokens and address tokens, with either alone failing on 25–35% of positives.**

### 11.4 Rare-token blocking (inverted index; ‘shares ≥1 token whose document-frequency ≤ T within the same-country S2∪S3 pool’)

Document frequencies are measured on the full same-country pool (US 6.19 M docs, India 4.13 M docs). Recall is on 120k positives per country; candidate counts are exact for 1,500 sampled S1 entities per country. ‘name_p4’ = 4-character token prefixes of the name (typo-tolerant); ‘addr’ = address tokens; ‘∪’ = union of the two candidate sets; ‘∧’ = must share both a name token and an address token.

**US**

| df ≤ T | recall name tokens | name_p4 | addr tokens | recall name_p4 ∪ addr | cands mean / p99 (∪) | recall name_p4 ∧ addr | cands mean (∧) |
|---|---|---|---|---|---|---|---|
| 20 | 5.1% | 1.5% | 10.4% | **11.8%** | 1 / 20 | 0.2% | 0.0 |
| 100 | 15.9% | 6.3% | 25.6% | **30.3%** | 16 / 111 | 1.5% | 0.1 |
| 300 | 34.5% | 15.2% | 43.8% | **52.4%** | 93 / 470 | 6.6% | 0.3 |
| 1000 | 40.8% | 26.1% | 68.6% | **76.8%** | 521 / 1,911 | 17.9% | 0.8 |
| 3000 | 57.3% | 45.4% | 86.2% | **92.6%** | 2,483 / 7,647 | 39.0% | 2.3 |
| 10000 | 72.2% | 66.3% | 93.8% | **98.1%** | — | 62.0% | — |

**India**

| df ≤ T | recall name tokens | name_p4 | addr tokens | recall name_p4 ∪ addr | cands mean / p99 (∪) | recall name_p4 ∧ addr | cands mean (∧) |
|---|---|---|---|---|---|---|---|
| 20 | 5.4% | 2.4% | 27.2% | **28.9%** | 4 / 22 | 0.7% | 0.0 |
| 100 | 21.8% | 10.1% | 44.6% | **50.1%** | 28 / 168 | 4.6% | 0.2 |
| 300 | 37.1% | 18.8% | 57.2% | **64.9%** | 124 / 600 | 11.1% | 0.5 |
| 1000 | 48.9% | 32.2% | 70.9% | **79.8%** | 594 / 2,233 | 23.3% | 1.2 |
| 3000 | 52.3% | 43.1% | 81.7% | **89.3%** | 2,444 / 8,110 | 35.5% | 2.9 |
| 10000 | 66.5% | 56.3% | 89.0% | **95.3%** | — | 49.9% | — |

**Take-aways.** (1) Tokens are *not* rare in this corpus: a real, correct match typically shares a token with df of a few hundred to a few thousand (names are composed from a limited vocabulary; addresses from a few thousand streets/localities per country). At df ≤ 300 the union blocker recalls only 52% (US) / 65% (India) of the positives with ≈90–125 candidates; **at df ≤ 3,000 recall is 93–94% (US) / 89–91% (India) but candidates rise to ≈2,400–2,700 per S1 (p99 ≈ 7,500–8,100)**; 98.1% (US) / 95.3% (India) is reached only at df ≤ 10,000 (candidate counts were not measured there and exceed the df≤3,000 figures). (2) **Address tokens are the stronger channel** (US 86% / India 82% at df ≤ 3,000 vs name 57% / 52%), consistent with the names being noisier and 100% of non-empty matched addresses sharing ≥1 token. (3) **Name prefixes recover typos** but add little on top of the address channel. (4) An AND-blocker (name ∧ address) is tiny (≈2–3 candidates) but only reaches 39% (US) / 36% (India) recall at df ≤ 3,000 (49% / 43% if full name tokens replace the 4-char prefixes) — a useful *high-precision tier* (it contains the ‘easy’ matches), not a complete blocker. (5) Union recall converges to 100% only when every token is allowed, which is the full pool again.

### 11.5 Score-based retrieval: top-K by IDF-weighted token overlap (name tokens + name 4-char prefixes ×0.5 + address tokens; tokens with df>20,000 ignored)

Instead of a hard token rule, candidates are ranked by Σ IDF of shared tokens and the top-K kept. Evaluated on 2,400 random S1 entities per country (*sample*; 2,263 US / 2,255 India of them have ≥1 true match):

| country | K | recall of true matches within top-K | entities with ALL their matches within top-K | test-scale pairs (1.73 M S1 × K) |
|---|---|---|---|---|
| US | 1 | 25.1% | 3.6% | 1,732,544 |
| US | 5 | 82.8% | 59.0% | 8,662,720 |
| US | 10 | 91.2% | 76.3% | 17,325,440 |
| US | 20 | 93.5% | 81.7% | 34,650,880 |
| US | 50 | 95.2% | 85.7% | 86,627,200 |
| US | 100 | 95.7% | 87.0% | 173,254,400 |
| India | 1 | 22.0% | 2.8% | 1,732,544 |
| India | 5 | 75.6% | 45.4% | 8,662,720 |
| India | 10 | 84.4% | 62.7% | 17,325,440 |
| India | 20 | 87.0% | 67.4% | 34,650,880 |
| India | 50 | 89.3% | 72.9% | 86,627,200 |
| India | 100 | 90.6% | 75.6% | 173,254,400 |

**Top-K by IDF overlap is far better than any hard rule**: K=20 already gives 93.5% (US) / 87.0% (India) match-level recall (vs ≈2,500 candidates for the df-cut union at 94%), and K=100 gives **95.7% / 90.7%**. The tail beyond K≈20 is flat (+2–4 points for 5× the candidates). The ceiling at K=100 costs ~2.8 points of macro-F0.5 even with a perfect classifier (oracle 0.9725). Recall is consistently ~5 points lower for India than for the US; consistent with (but not proven by) native-script names combined with reordered/abbreviated long addresses — this is the first thing to dissect when building the real retriever. The retrieval step in this unoptimised sparse-matrix implementation measured ≈ 45 ms (US) and ≈ 78 ms (India) per S1 entity on one core → for the 1.73 M test entities ≈ 30 core-hours single-threaded, i.e. ≈ 3 h on this 10-core machine before any optimisation (§17).

### 11.6 Blocking conclusions (what the data supports)

- **Country equality → then a *union* of retrieval channels**, because no single channel exceeds ~85% recall: (a) address-token IDF, (b) name-token/prefix IDF, (c) an exact-key tier (`sorted` address or `sorted` name equality; 47% of positives are found by one of them, §8), plus for the ~18% native-script India names an address-only channel. The channels’ union at moderate K is what reaches ≥93–95%.
- **Rare name tokens**: useful, but ‘rare’ means df of a few hundred–few thousand, not tens. **Address tokens** are the most reliable signal (rare-token address channel alone: 86% recall at df ≤ 3,000). **Postal/PIN components: not available** (§7) — the closest substitutes are the numeric ‘house/plot’ token and locality tokens. **City/state tokens** are nearly useless as blockers on their own: state ≈ 1/50 of the US pool (272k candidates), and in France ~15 cities cover most records.
- **Name + address combinations** are the way to get *small* candidate sets: AND tier (~2–3 candidates, ≈35–49% recall) ⊂ top-K ranked union (K≈20–50, 87–94% recall).
- **Where recall is genuinely lost** (§9): empty matched address (4.4%), native-script/unrelated names, and mutated house numbers. Roughly 1% of positives have neither a usable name nor a usable address.
- Scale to plan for: **K = 20–50 → 35–87 M candidate pairs for test**, K = 100 → 173 M. All are comfortable for a tree model in chunks; the retrieval step is the expensive part.

## 12. SINGLETON ANALYSIS

**Singleton rate: 123,247 / 2,206,821 = 5.585%** (US 5.58%, India 5.59% — identical). Because macro-F0.5 gives a singleton **1.0 for predicting nothing and 0.0 for predicting anything**, this 5.6% of entities is worth 5.6 points of the final score and is the only group where a false positive costs a full 1.0.

**Do singletons have distinctive names/addresses?** No. Every S1-side attribute is identical for singletons and non-singletons to within sampling noise:

| attribute (mean) | non-singleton | singleton |
|---|---|---|
| name length (chars) | 24.036 | 24.003 |
| address length (chars) | 52.066 | 52.071 |
| name tokens | 3.625 | 3.621 |
| # other S1 with same normalised name | 10.239 | 10.255 |
| # S1 at same normalised address | 1.139 | 1.14 |
| rarest name-token document frequency | 533.268 | 532.937 |
| has legal suffix | 0.651 | 0.651 |

Singleton rate by name multiplicity (1 / 2 / 3–5 / 6–20 / 21+ entities sharing the name): 5.58% / 5.52% / 5.64% / 5.58% / 5.59%; by address multiplicity: 5.58% / 5.64% / 5.62%; by name length bucket: 5.63% / 5.66% / 5.58% / 5.58% / 5.56% / 5.48%. **Singletons are a uniformly random 5.6% of S1 entities**: no S1-side feature predicts them, so the decision ‘no match’ can only come from *the absence of convincing candidates*, never from the entity’s own characteristics. (Examples of real singletons look like every other entity: `Beth Presbyterian Church | 1230 Palmer Street, Downers Grove, IL`, `Ace Systems Private Limited | Lucknow, Seemant Nagar, …`.)

**Are they likely to create dangerous false positives? Yes — measured.** Using the realistic retrieval of §11.5 (top-100 candidates) and a small diagnostic gradient-boosting scorer (trained on 1,200 entities/country, evaluated on 1,200 held-out entities/country; *diagnostic instrument*, not the final model), the share of entities that have **at least one false candidate scored above a threshold p**:

| entities | n | max false-candidate p ≥ 0.5 | ≥ 0.8 | ≥ 0.9 | ≥ 0.95 |
|---|---|---|---|---|---|
| singletons (all candidates are false) | 1,354 | 23.6% | 11.9% | 7.3% | 4.8% |
| non-singletons (their non-matching candidates) | 2,243 | 14.5% | 6.7% | 4.1% | 2.3% |

So **≈ one singleton in four (24%) has a false candidate the diagnostic scorer rates ≥ 0.5, and 4.8% have one ≥ 0.95** — 1.6–2.1× the rate for non-singletons’ wrong candidates (which are diluted by having true matches to outrank them). Effect of the decision threshold on the sampled macro-F0.5 (this diagnostic scorer):

| threshold p | macro F0.5 (random 2,400-entity sample) | US | India | singletons | non-singletons |
|---|---|---|---|---|---|
| 0.3 | 0.8744 | 0.9216 | 0.8272 | 0.6417 | 0.8917 |
| 0.5 | 0.8893 | 0.9345 | 0.8442 | 0.7624 | 0.9013 |
| 0.6 | 0.8933 | 0.9393 | 0.8473 | 0.8036 | 0.9029 |
| 0.7 | 0.8926 | 0.9397 | 0.8454 | 0.8501 | 0.8981 |
| 0.8 | 0.8884 | 0.9355 | 0.8413 | 0.8816 | 0.8919 |
| 0.9 | 0.8754 | 0.9266 | 0.8241 | 0.9273 | 0.8742 |
| 0.95 | 0.8532 | 0.9132 | 0.7932 | 0.9528 | 0.8485 |

The overall optimum is flat around p≈0.6–0.7 (0.893); raising the threshold to 0.95 lifts singleton score from 0.76 (p=0.5) to 0.95 but costs the non-singletons ~5 points. **The singleton/non-singleton trade-off is real and must be optimised explicitly on grouped validation data, not fixed at 0.5.** For scale: a false match on a 3-match entity lowers its F0.5 from 1.0 to 0.789, a missed match to 0.909 (a false positive costs ~2.3× a false negative), while a false match on a singleton costs 1.0.

**Where the dangerous false positives come from** (inspected: the 162 singleton false candidates with p ≥ 0.8 in the evaluation samples — 130 (80%) are unmatched decoys and 32 (20%) belong to another S1 entity):
1. **Sibling look-alike records — unmatched decoys (80% of the high-scoring ones; US 68%, India 85%).** Same building/complex/street and a shared name stem but a different unit/plot number and one different name word: `Goodwill Academy | 160/218, Ajmeri Gate` vs `Goodwill Academy Exports Corp | 160/225, …`; `Ys Welfare Society | 331/9K` vs `Ys Welfare Society Private Limited | 331/12K`; `Innovative Bharat Properties Private Limited | 1/19` vs `INNOVATIVE BHARAT LOGISTICS PRIVATE LTD | 1/30`; `Kaykay Sarkar Ltd | 37/26` vs `Kaykay Sarkar Group | 37/37`; `Technologies Poineer Customer Private Limited` vs `… Public Limited`. (Only 3% of these decoys have an identical normalised name and <1% an identical address.) One cheap discriminator exists: a **conflicting legal form** (`Private` vs `Public`, `LLC` vs `Inc`) occurs in 5.4% of high-scoring false candidates vs 0.41% of true pairs (§6). These are exactly the records that a name-and-address similarity model *will* accept — and the discriminating evidence (a different plot number, a different suffix word) is the same kind of perturbation the noise generator applies to *true* matches (house number differs in ~17% of positives; suffix words are added/dropped/changed in ~35%). This is the central precision problem of the task.
2. **Records belonging to *another* S1 entity (20%; US 32%, India 15%).** Multiple S1 entities at the same address (5.3% of S1 rows share an address; co-working towers, `Devya Organic Pvt. Ltd.` vs a differently named company in the same plot) — especially when the candidate’s name is in native script (`లక్ష్మీ ప్రొడ్యూసర్…`, `स्काई इंडस्ट्रीज…`), so the name cannot veto the address match. Because ground truth assigns every S2/S3 record to at most one S1 entity (§5), a one-to-one assignment step (a record goes to the S1 entity it fits best) is available to suppress this class.
3. Records with an empty address are 97.8% likely to be matched to *some* entity (§13), so they are rarely decoys; their risk is instead being assigned to the wrong entity when only a short or native-script name is available.
Decoy prevalence overall: 26.6% of S2 and 25.4% of S3 training records match no S1 entity, yet only **0.97% of unmatched S2 addresses occur in S1** (vs 30.4% of matched) and 5.3% of unmatched names equal an S1 name (vs 31.7% of matched) — so *most decoys are unrelated entities*, and the dangerous minority is a small, structured subset that survives blocking because it looks like the truth. **Test appears to contain proportionally more decoys** (§14), so the precision side matters more there.

## 13. DATA QUALITY

| issue (row counts) | train S1 | train S2 | train S3 | test S1 | test S2 | test S3 |
|---|---|---|---|---|---|---|
| names with double space | 0 | 554,392 | 575,616 | 0 | 500,237 | 519,596 |
| names with control chars | 1 | 3 | 1 | 0 | 0 | 0 |
| addresses with control chars (0x1A / 0x7F) | 70 | 119 | 125 | 72 | 183 | 162 |
| addresses with mojibake (`Â\x80\x93`, `Â\x80\x99`) | 443 | 1,118 | 785 | 400 | 1,239 | 856 |
| addresses with HTML entity (`&#65533;`) | 1 | 3 | 0 | 1 | 1 | 3 |
| addresses with double space | 968 | 107,574 | 4,874 | 753 | 82,739 | 3,126 |
| addresses > 150 chars | 2,032 | 2,883 | 2,277 | 1,829 | 3,309 | 2,589 |
| addresses > 300 chars | 0 | 0 | 0 | 0 | 0 | 0 |
| names ≤ 2 chars | 0 | 704 | 9,275 | 0 | 10,166 | 19,910 |
| names ‘NA’/‘null’-like | 0 | 6 | 18 | 0 | 49 | 61 |
| names only digits/punctuation | 0 | 194 | 423 | 0 | 111 | 238 |
| leading/trailing whitespace (name or address) | 0 | 0 | 0 | 0 | 0 | 0 |

Findings:

- **IDs:** all 12.5 M train + 11.7 M test IDs are well-formed (`S<k>-<digits>`), **0 malformed, 0 duplicated within a file, 0 shared between train and test**; numeric parts are random 1–9-digit integers (so ID length ≈ uniform; 81.7% of matched S1/match ID pairs have equal digit-length vs 81.8% expected by chance).
- **Empty strings:** the only empty field is `business_address` — S2 168,967 (3.36%), S3 175,916 (3.33%) in train; test S2 129,408 (2.65%), S3 136,098 (2.68%); S1 never. Empty-address records are almost always *matched* records (§ below), i.e. missingness is injected noise, not natural sparsity.
- **Placeholder tokens inside addresses** (not empty): `NULL`, `<NULL>`, `N/A` appear in 175,568 S2 and 174,742 S3 train addresses (≈3.4%) but 79 in S1 — they must be stripped before tokenising or `null` becomes a high-df token that ‘matches’ everything.
- **Pandas default NA parsing corrupts data:** names such as `NA`, `Na`, `NAN`, `null`, and any address `N/A`/`NULL` would silently become `NaN`; reading with `keep_default_na=False, dtype=str` avoids this. Use `quoting=csv.QUOTE_NONE` (stray `"` in 4–349 rows/file).
- **Control characters / encoding damage (small but real, and present in S1 too, i.e. from the source data rather than the noise generator):** `\x1A` (SUB) has replaced apostrophes/special characters (`Shopper\x1aS Stop`, `D\x1asouza Colony`, name `Medchal\x1aMalkajgiri Housekeeping Private Limited` appears 1/3/1 times in S1/S2/S3 — the same entity), 70/119/125 train addresses; UTF-8 en-dash/apostrophe mis-decoded as Latin-1 (`Â\x80\x93`, `Â\x80\x99`) in 443/1,118/785 train addresses (0.02% of rows; these are the U+0080–U+0099 ‘Cc’ characters); `&#65533;` numeric HTML entity for the replacement char. There are **no U+FFFD characters, no BOMs, no tabs/newlines inside fields, and all non-ASCII names in a 30k-name sample were already NFC**; in a 200k-row sample per file the only odd characters of note are ZERO WIDTH NON-JOINER (U+200C, inside Indic conjuncts), `°` (degree sign in addresses), typographic apostrophe U+2019, and the C1 controls U+0080–U+0099 from the mojibake.
- **Extremely long fields:** names ≤ 123 chars; addresses ≤ 269 chars (p99.9 ≈ 150); 0.09% of S1 and 0.04–0.06% of S2/S3 addresses exceed 150 chars (all Indian multi-line addresses with repeated district/locality names, e.g. `Visakhapatnam, Visakhapatnam, Visakhapatnam, Vishakhapatnam`). Nothing >300.
- **Very short / junk names:** S3 has 9,275 names of ≤2 characters in train (`SC`, `SI`, `SS`, `GM` …; mostly India 7,799), S2 only 704. **In test the pattern moves to France:** test S2 has 10,166 (9,748 France = 1.39% of France S2 rows vs 0.014% of train S2) and S3 19,910 (12,182 France). These 2-letter names look like initials of the real name (`CC`, `AF`, `BG`) — **no name evidence at all** — and are a new, France-specific S2 noise type absent from train (§14). Also 194–423 names are only digits/punctuation (`18  29`, `1 800`).
- **Suspiciously repeated names/addresses:** in S1 45–46% (India) / 36% (US) of rows share a name with ≥1 other S1 row (max 253 for `Primary Care Group`); 5% share an address. S2/S3 top repeated names are the generic `Primary Care`, `Physical Therapy`, `Womens Health`, `earnosethroat.com`, `CC`. Shared addresses in S3 India include truncated addresses like `Ground Floor, Bangalore, KA` (29 records), `303, Mumbai, MH` — addresses stripped to a floor/door number and a city, which are *not identifying*.
- **Exact duplicate rows inside S2/S3:** S2 has 50,933 rows in 25,060 groups of identical (name, address, country); S3 37,241 rows / 18,381 groups. **100% of these rows are matched, and in 100% of the groups all rows are matched to the same S1 entity** (0 mixed, 0 unmatched). An exact in-source duplicate is therefore a certain positive-cluster signal (a free consistency constraint, and a leakage-like structure to be aware of).
- **Record-level artefacts that predict ‘matched’ (measured on train S2; S3 is the same):**

| record feature | unmatched (decoy) S2 | matched S2 | P(matched | feature) |
|---|---|---|---|
| address empty | 0.28% | 4.47% | **97.8%** |
| name contains `.com` | 0.56% | 5.24% | **≈ 96%** |
| mean name length (chars) | 28.6 | 23.8 | shorter ⇒ more likely matched |
| mean name tokens | 4.04 | 3.30 |  |
| name ALL-CAPS | 14.3% | 20.5% |  |
| country = US | 59.9% | 59.9% | no difference |
| ID length / ID value / row position (KS distance) | — | — | KS 0.005 / 0.005 / 0.005 → no signal |

Decoys are *cleaner and longer* than matched records (matched records have been through the noise generator: truncated names, `.com` forms, blanked addresses; decoys were generated as complete standalone entities). This is a legitimate record-level feature at test time, but it (a) inflates any validation that treats ‘noisy = match’, and (b) may not transfer to France, where the noise mixture differs (initials-only names).

## 14. TRAIN / TEST COMPARISON

| statistic | train | test | test/train |
|---|---|---|---|
| S1 rows | 2,206,821 | 1,732,544 | 0.79× |
| S2 rows | 5,034,616 | 4,887,273 | 0.97× |
| S3 rows | 5,285,603 | 5,082,316 | 0.96× |
| S2 records per S1 entity | 2.28 | 2.82 | 1.24× |
| S3 records per S1 entity | 2.40 | 2.93 | 1.22× |
| S1 name length mean / p95 / max | 24.03 / 37.0 / 105 | 23.84 / 36.0 / 92 |  |
| S2 name length mean / p95 / max | 25.1 / 40.0 / 104 | 25.7 / 42.0 / 102 |  |
| S3 name length mean / p95 / max | 25.2 / 42.0 / 123 | 25.66 / 42.0 / 103 |  |
| S1 address length mean / p95 / max | 52.07 / 103.0 / 256 | 57.21 / 105.0 / 268 |  |
| S2 address length mean / p95 / max | 46.23 / 96.0 / 249 | 50.41 / 99.0 / 269 |  |
| S3 address length mean / p95 / max | 46.71 / 91.0 / 240 | 48.74 / 94.0 / 267 |  |
| S2 empty address | 3.36% | 2.65% |  |
| S3 empty address | 3.33% | 2.68% |  |
| S1 names non-ASCII | 0.0% | 2.4% |  |
| S1 names with double space | 0.0% | 0.0% |  |
| S2 names non-ASCII | 15.2% | 19.0% |  |
| S2 names with double space | 11.0% | 10.2% |  |
| S3 names non-ASCII | 11.5% | 14.5% |  |
| S3 names with double space | 10.9% | 10.2% |  |
| mean address tokens S1 / S2 / S3 | 8.03 / 7.29 / 7.17 | 8.59 / 7.8 / 7.51 |  |

**Countries:** train US 60.0% / India 40.0%; test US 38.3% / India 46.8% / **France 15.0%** of S1 (14.4% of S2/S3). Per-country S1 structure for the two shared countries is **essentially identical** to train — India 45.3% of S1 rows share a name (train 46.0%), US 29.2% (train 35.9% — the lower US share is the expected size effect of a smaller S1 pool: 663k vs 1.32M entities); India name length 26.4 (26.4), address 77.7 chars (77.7), 11.2 tokens (11.2); US name 22.5 (22.5), address 35.0 (35.0), 5.9 tokens (5.9); legal-suffix mix identical (India `Private Limited` 73.1% vs 73.3%; US `LLC` 50.2% vs 50.0%, `Incorporated` 33.8% vs 33.8%). The generator is unchanged for US/India; the shift is (a) the *mix* of countries, (b) France, (c) a larger S2/S3 pool relative to S1.

**Distribution shift to pay attention to**

1. **France is an unseen domain, ≈15% of test entities** (≈260k S1 entities; 1.43 M S2/S3 records). 53% of French S1 name tokens and 37% of French address tokens never occur in any training file (vs 1.1–1.5% for the same measurement on US/India test names). French names are short (mean 19.4 vs 24 chars), template-like (`Bordeaux Club SARL` appears 205 times as an S1 name; `Nantes Club SARL` 157) and 34% of French S1 rows share their name with another entity; **8.5% share their address (train ≈5%; up to 99 entities at `12 RUE Lyderic, Lille`)**, so duplicates-with-different-entity (the twin problem) is likely *worse* in France. There are 15.7% accented names in S1 (train S1: 0%), French legal forms (`SARL/SAS/EURL/SASU/SCI/SA`, S2/S3 also `S.A.S`, `S.A.R.L`), abbreviated street types (`R.`, `Av`, `Bd`, `Pl.`, and `Allée`/`Allee` spellings), no postal codes, and a small set of ~15 cities — geography is uninformative.
2. **Larger decoy share in test.** Test has 2.82 S2 and 2.93 S3 records per S1 entity vs 2.28 / 2.39 in train (+24% / +23%). Independent check: the share of S2/S3 records whose (token-sorted) address also occurs in S1 is **~20% lower in test** at the same country (S2 US 23.6% vs 29.3%; S2 India 9.4% vs 11.9%; S3 US 4.0% vs 4.9%; S3 India 3.6% vs 4.5% — a uniform ≈0.80× factor), while the share of *S1* entities that have an exact-address hit in S2∪S3 is unchanged (US 46.9% vs 47.5%; India 29.3% vs 29.5%). Both point the same way: **similar matches per S1 entity, but ≈ 41% of test S2/S3 records unmatched vs ≈ 26% in train** (solving `share = m·a + (1−m)·0.0097` with the train-calibrated matched-record rate `a`, per country, gives m ≈ 0.59 for both US and India) (inference from unsupervised proxies, not a label; if true, the false-positive pressure per S1 entity is higher in test and precision-oriented thresholds tuned on train will be slightly too permissive).
3. **New noise type in France S2:** 1.4% of France S2 records (and 1.7% of France S3) have 2-letter, initials-like names vs 0.014% in train S2. Name uppercase rates differ too (France S2 names all-caps 20.8%, S3 5.6% — same style split as train).
4. **No entity/ID leakage between splits:** 0 shared IDs; 0 test S1 rows whose (normalised name, address) pair exists in train S1. Names *are* reused across the split (35% of test S1 normalised names occur in train S1: India 46%, US 36%, France 0.01%) because names are template-generated, so name-frequency features fitted on train will partly transfer for US/India and not at all for France; addresses rarely repeat (5% of test S1 addresses occur in train S1).
5. **Vocabulary:** US/India test tokens are only slightly more OOV than a held-out train baseline (name tokens 1.5% vs 1.15% US, 1.1% vs 0.75% India; address tokens 1.9% vs 1.5% US, 2.9% vs 2.1% India) — i.e. no vocabulary shift for the shared countries beyond the ~+0.4–0.8-point out-of-sample effect.


## 15. RECOMMENDED BASELINE (not implemented — derived only from the observations above)

**Constraint reminder from the README:** `candidate_pairs.tsv` must be *exactly* the set the classifier scores, so the retriever’s K directly sets that file’s size (K=50 → ≈87 M IDs ≈ 1 GB); the final model must be MIT/Apache-2.0 and ≤8 B parameters (gradient-boosted trees satisfy this trivially).

**Step 0 — Loading.** `pd.read_csv(sep='\t', dtype=str, keep_default_na=False, na_filter=False, quoting=csv.QUOTE_NONE)`; convert once to parquet; keep IDs as (source, int) pairs. *(Why: names `NA`/`Na`/`null` and empty addresses are real values; 134–349 lines per test file contain a bare `"`.)*

**Step 1 — Normalisation (per field, deterministic; country-aware dictionaries, global fallback).**
- *Cleaning:* map `\x1A`/`\x7F` → space, repair `Â\x80\x93/\x99` → `-`/`'`, drop `&#65533;`, NFKC, casefold, fold Latin diacritics only (never Indic combining marks), punctuation → space, collapse whitespace.
- *Names:* strip `(ID: 12345)` / trailing `#12345`, leading junk punctuation and honorifics (`Mr/Mrs/Shri/Smt/Sri/M/s`); resolve **alias markers** (`dba|d/b/a|f/k/a|fka|a/k/a|aka|t/a|trading as|formerly`) by keeping the text *after* the marker as the core name (the true S1 name is there in 99.9% of alias cases) and the text before as a weak secondary field; turn `xxx.com`/`www.` into its stem; **pull the legal form out of the name** (`pvt|private|ltd|limited|llc|l.l.c|llp|inc|incorporated|corp|corporation|co|company|pllc|pc|sarl|sas|sasu|eurl|sci|sa…` → canonical type) and compare *core names* and *legal types separately* (a dropped suffix is the most common difference, 22%, and must not be penalised like a mismatch — but a *conflicting* legal form is 13× enriched among false candidates); `&`↔`and`; keep both ordered and token-sorted forms (6% pure word-order changes; suffix words move to the front/middle).
- *Addresses:* drop `NULL/<NULL>/N/A`; move `PO BOX/PMB/Unit/#/Apt/Floor` tokens into a separate ‘secondary’ field; strip leading zeros from numbers (`0014048` → `14048`); extract the **first house/plot number token(s) as their own feature** (equal in 79%, 83% after zero-strip); expand street-type abbreviations with a dictionary (`rd/road, st/street, ave, dr, ln, ct, cir, blvd, hwy…`); map **state code ↔ full name ↔ native-script name** and India district/city variants (`Madras/Chennai`, `Bangalore/Bengaluru`, `Ahmadabad/Ahmedabad`) with equivalence tables **mined from aligned training positives** (S1 last component vs S2/S3 components) rather than hand-typed; sort components (5–8% pure reorders).
- *France (no labels):* accent folding, `R./Rue`, `Av/Avenue`, `Bd/Bvd/Boulevard`, `Pl./Place`, `Allée/Allee`, `S.A.S/SAS`, `S.A.R.L/SARL`, treat `de/du/des/la/le` as stop-tokens (they are the top address tokens) — derived from the test S1/S2/S3 text distributions, not from train.

**Step 2 — Blocking (within `country` string equality; union of channels; top-K by IDF score).** (a) *exact tier*: equal token-sorted normalised address, or equal token-sorted normalised name (finds ≈47% of positives and is very precise *when the address/name is not shared by several S1 entities*); (b) *address channel*: sparse IDF-weighted token overlap; (c) *name channel*: IDF-weighted name tokens + 4-char prefixes (typo tolerance); (d) for records with a native-script name only the address channel applies. Rank by summed IDF, keep K≈30–50 per entity; expected match-level recall from the simple version measured here: **≈ 93–96% US, ≈ 87–91% India at K = 20–100** (oracle macro-F0.5 ceiling 0.97 at K=100). IDF must be **country-specific** (a token like `bordeaux` or `rue` has a different df in each pool). Do not use ZIP/PIN (absent), state alone, or first-token keys as the sole blocker.

**Step 3 — Pair features (≈30–40, computed on Step-1 fields).** *Name:* `token_sort/set/partial ratio`, Jaro-Winkler, normalised Levenshtein, token Jaccard/overlap, **char-3-gram TF-IDF cosine** (best single name feature: AUC 0.987 on Latin names), core-name similarity, legal-type agreement and a **legal-form *conflict* flag** (13× enriched among dangerous false candidates), first-token equality, length/token-count deltas, flags for native-script / domain-form / alias / initials-only (≤2-char) names, ‘S1 name shared by n other S1 entities’. *Address:* token-set ratio, Jaccard, IDF-weighted overlap, char-3-gram TF-IDF cosine, **house-number equal / equal-after-zero-strip / off-by-one-edit**, state agreement, city/locality token agreement, missing/extra component counts, empty-address flag. *Context:* source (S2 vs S3 — their noise styles differ), country, retrieval rank/score/relative score/gap to next candidate, **competition features** (number of S1 entities for which this S2/S3 record is a candidate; whether this S1 is the record’s best-scoring S1).

**Step 4 — Classifier.** Gradient-boosted trees (LightGBM/XGBoost, or scikit-learn `HistGradientBoostingClassifier`, which already gave AUC 0.998 on sampled pairs and macro-F0.5 ≈ 0.89 in the end-to-end diagnostic below). Missing address → native `NaN`. **Train on the retriever’s candidate lists (with decoys and sibling look-alikes), not on random negatives** — random negatives are trivially separable (AUC 0.91–0.99 per single feature) and would over-state precision; positives are ≈3–4% of a top-100 list.

**Step 5 — Decision rule.** (i) score each candidate; (ii) enforce the **one-to-one property** observed in ground truth (each S2/S3 record ↔ at most one S1): a record is offered only to the S1 entity where it scores highest (drop on near-ties); (iii) predict candidates with p ≥ τ, tuned by **macro F0.5 over grouped validation entities including singletons** — in the diagnostic the optimum was flat at τ≈0.6–0.7; (iv) keep exact in-source duplicates (identical name+address+country within S2 or S3) together — they always share one S1 owner; (v) return an empty list when no candidate clears τ.

**Step 6 — Validation protocol.** Split **by S1 entity** (all its matches in the same fold; names are template-generated so row-level splits leak), evaluate on candidate lists built from the *full* S2/S3 pool (decoys included), report macro-F0.5 separately for US/India, S2/S3, singleton/non-singleton and native-script/Latin; **leave-one-country-out (train US → test India and vice-versa)** as the only available proxy for the unseen France domain.

**What this baseline should be expected to reach:** the end-to-end diagnostic (K=100 IDF retrieval + 19 features + boosted trees trained on only 1,200 entities/country, no one-to-one rule, no context features) scored **macro-F0.5 ≈ 0.89** on held-out entities (US 0.94, India 0.85; singletons 0.80–0.85 at the optimal τ). A properly built version of Steps 1–5 should do materially better; the gap to the 0.97 oracle ceiling is concentrated in India, in singletons (sibling look-alikes) and in native-script/alias names.

## 16. RECOMMENDED ADVANCED PIPELINE — what to investigate next (and what the data says *not* to spend time on)

| technique | verdict from the data | why / what to test |
|---|---|---|
| Fuzzy string matching (RapidFuzz etc.) | **Yes — core** | Most positives need fuzzy comparison (only 4.6% identical names). Token-set/partial ratio and Jaro-Winkler are strong; plain Levenshtein is the weakest name feature (AUC 0.90 vs random) and fails on order changes. |
| TF-IDF (word and char n-gram) | **Yes — for retrieval *and* features** | Char-3-gram TF-IDF is the best single name feature (AUC 0.987 Latin) because typos/leet/accents break tokens but not trigrams; address TF-IDF is the best single address feature (0.956). Fit IDF per country on S2∪S3. |
| Character n-grams / MinHash-LSH | **Yes as a second retrieval channel** | Complements token retrieval for typo-heavy names; candidate to close part of the India recall gap (90.7% @100). Needs a native-script-aware analyzer. |
| Sentence/multilingual embeddings | **Not as the main matcher; test narrowly for native-script names** | Name noise is orthographic (typos, order, suffix), not semantic, and names are built from a small template vocabulary — semantic similarity would merge siblings (`Goodwill Academy` / `Goodwill Academy Exports`). The one place they may pay off is **Devanagari/Telugu/Tamil/… ↔ Latin** (18% of India positives; 9% of S2, 5% of S3 names). Compare against a **learned token-level translation table mined from the 7.6 M training pairs** (cheaper, exact; India OOV is only ≈1.1%). Any model must be MIT/Apache and ≤ 8 B. |
| GBDT (XGBoost / LightGBM / HistGB) | **Yes — first model** | Heterogeneous numeric features, ~4–5% missing address, class imbalance in candidate lists, need for calibrated probabilities for an F0.5 threshold. |
| Second-stage / listwise model | **Yes** | Aggregate over an entity’s candidate list (score gaps, #candidates above τ, competitors at the same address, one-to-one conflicts, S2–S3 mutual agreement). This is where sibling decoys and same-address entities are resolved. |
| Ensembles | Later | Country-specific vs global GBDTs, name-only + address-only + joint models; likely small gains over good features. |
| Threshold optimisation | **Yes — high value** | Macro-F0.5 with 5.6% singletons: tune τ per source (S2/S3), per evidence type (native-script name, empty address), and consider choosing each entity’s predicted set to maximise *expected* F0.5 given candidate probabilities (≤11 matches; a false match costs ≈2.3× a miss, and 1.0 on a singleton). |
| Source-specific handling | **Yes** | S2 and S3 differ systematically (S2: ALL-CAPS, reorders, drops components, no DBA aliases, 12.5% exact-normalised address matches; S3: spelled-out US states, India 2-letter states, DBA aliases, 4.3% exact addresses). Add `source` as a feature at minimum; test separate models. |
| Country-aware processing | **Yes, with a global fallback** | Separate normalisers/IDF/threshold *tables* keyed by the country string; no one-hot on {US, India}. Validate transfer with leave-one-country-out. For France: unsupervised dictionaries from test text (check the competition rules on fitting unsupervised statistics on test data before doing this; document it). |
| Graph / cluster consensus | **Worth testing** | S2/S3 records of one entity are similar to each other (and exact in-source duplicates always share an owner); use S2↔S3 similarity as evidence for borderline candidates and to break one-to-one conflicts. |
| Postal-code / geocoding features | **No** | No ZIP/PIN/postcodes exist in any file; geocoding APIs are prohibited. |
| Global neighbour-graph or LLM matching | No (too expensive at 10⁷ records; ≥8 B limits) | Pair volume (35–170 M scored pairs) favours cheap features + trees. |

**Concrete next investigations (ordered by expected value):** (1) dissect the India recall gap at K=100 (native names? reordered long addresses? mutated numbers?); (2) mine abbreviation/state/city/transliteration equivalence tables from aligned positives and measure coverage on test tokens; (3) build features that separate **sibling decoys** from true matches with mutated house numbers (unit-number conflict × suffix-word conflict × source); (4) implement the one-to-one assignment and measure how much of the singleton false-positive rate it removes; (5) quantify France: token-frequency profile, expected duplicate-name/address ambiguity, initials-only names, and whether the exact tier alone yields a high-precision first pass for pseudo-checking; (6) calibrate probabilities and choose τ by grouped-validation macro-F0.5.

## 17. MEMORY / PERFORMANCE

| item | measured / estimated |
|---|---|
| Raw TSV size | train 1.20 GB (S1 0.21 + S2 0.49 + S3 0.50) + ground truth 0.13 GB; test 1.19 GB (0.18 + 0.51 + 0.51); total ≈ 2.5 GB (decimal) |
| In pandas RAM (object strings) | train S1 0.63 GB, S2 1.58 GB, S3 1.63 GB, GT 0.34 GB; test S1 0.52 GB, S2 1.61 GB, S3 1.62 GB — **≈ 8.0 GB for all seven** |
| Fits in 16 GB? | **Yes for loading, no for everything at once with derived columns.** Load once → parquet (2–6 s per file), process **per country** and per source, use `string[pyarrow]`/categoricals for `country`, int64 for ID numbers. Peak in my retrieval experiments ≈ 5 GB. |
| Parsing | TSV → pandas 2–6 s per file (C engine, `dtype=str`); no chunking needed for parsing |
| Normalisation (`ext`, regex with callbacks) | ≈ 40–65 s per 5 M-record source for name + address (measured 27 s for the 2.2 M-record S1); chunk at 0.4–1 M rows |
| Vectorising (CountVectorizer, 3 views) | ≈ 30 s per country over the 4–6 M-record pool; sparse matrices 15–42 M non-zeros per view (≈ 0.2–0.5 GB each) |
| Token retrieval (sparse product, top-K) | **≈ 45 ms (US) / 78 ms (India) per S1 entity on one core** → ≈ 30 core-hours for 1.73 M test entities naive; ≈ 3 h on 10 cores. Needs batching (100–500 entities), df cut-off (20k), float32, `multiprocessing` per country/chunk — or an exact-tier-first design that only retrieves for unresolved entities |
| Pair features (RapidFuzz + Python) | ≈ 58 µs/pair for 19 features (600k pairs in 35 s) → K=50: 87 M pairs ≈ 1.4 core-hours; K=100: 173 M ≈ 2.8 core-hours (≈ 17 min on 10 cores). Vectorise with `rapidfuzz.process.cdist` per chunk; never `df.apply(axis=1)` |
| Feature matrix | 173 M pairs × 40 float32 = 28 GB → **must stream** in 1–2 M-pair chunks (score, keep p and IDs, discard features) |
| Brute-force pair space | test 1.7×10¹³ (6.7×10¹² same-country); train 2.3×10¹³ (1.2×10¹³ same-country) — never enumerate |
| Realistic candidate scale | K = 20 → 35 M; 30 → 52 M; 50 → 87 M; 100 → 173 M test pairs; the exact-address/name tier adds well under 1 pair per entity |
| Expensive vs cheap | Cheap: exact-key joins, groupby, value_counts, ground-truth explode (7.6 M pairs, seconds). Expensive: retrieval matmul, per-pair fuzzy features, regex-callback normalisation, TF-IDF fitting on all 10 M records (use a 0.5–1 M sample). |

## 18. FINAL EXECUTIVE SUMMARY

**DATASET SIZE:** Train: 2,206,821 S1 / 5,034,616 S2 / 5,285,603 S3 records, 7,638,365 true pairs. Test: 1,732,544 S1 / 4,887,273 S2 / 5,082,316 S3 records (≈ 11.7 M records, 1.19 GB; every S1 needs a row). 2.5 GB of TSV in total; fits in 16 GB RAM if processed per country. Clean file format (0 malformed rows, 0 duplicate IDs, valid UTF-8).

**MAIN NOISE TYPES:** Names — case/whitespace/punctuation (≈17%), abbreviation & legal-suffix swaps/drops (`Ltd/Limited`, `Corp/Corporation`, `Co/Company`, `Inc→Incorporated`, `LLC→L.L.C.`; suffix dropped in 22% of pairs; note `Pvt`↔`Private` is *listed* in the task but **never flips** in the data), word-order shuffles (6%), typos and digit-for-letter swaps, diacritic injection, honorifics/fake `(ID: n)` suffixes/junk prefixes/domain-style names (`xxx.com`), DBA/F/K/A aliases (S3), duplicated/added/dropped words (~25%), **full native-script rewrites (18% of India pairs)** and unrelated aliases/2-letter initials (~4%). Addresses — ALL-CAPS and abbreviations (S2), spelled-out US states / 2-letter India states (S3), reordered components, `NULL/N/A` fillers, `PO BOX/PMB`, zero-padded and **mutated house numbers (17% of pairs)**, dropped/misspelled/substituted cities (11–12%), missing state (15% of S3 US), native-script states, empty address (4.4%). **No ZIP/PIN codes anywhere.**

**MATCH DENSITY:** 3.46 matches per S1 entity (3.67 excluding singletons), median 3, max 11; S2 1.67 + S3 1.79 per entity; 80.5% of entities have matches in both sources; only 5.4% have exactly one. 73–75% of S2/S3 records are matched (each to exactly one S1 — a strict partition), 26% are decoys. Test has ≈ 24% more S2/S3 records per S1 and appears to have ≈ 41% decoys.

**SINGLETON RATE:** 5.585% (123,247), identical in US and India, statistically unrelated to any S1 attribute. A false match on a singleton costs 1.0 of that entity’s score (≈ 2.3× a false negative elsewhere). In the diagnostic, 24% of singletons receive a false candidate scored ≥ 0.5 and 4.8% one ≥ 0.95 — mostly engineered sibling look-alikes (same building, near-identical name, different unit number/suffix word).

**MOST IMPORTANT SIGNALS:** (1) **Address token overlap** — every non-empty matched address shares ≥ 1 token; address TF-IDF AUC 0.956, and it is the only signal separating same-name entities (38% of S1 rows share a name). (2) **Name char-n-gram/token similarity** — AUC 0.987 on Latin names but only 0.83–0.86 overall (native/alias names). (3) **House/plot number equality** (79–83%). (4) **Country equality** (100% of positives). (5) Source identity (S2 vs S3 noise styles). (6) Structure: one-to-one partition, in-source exact duplicates ⇒ same owner, empty address ⇒ 97.8% matched. Name and address are complementary: either ≥ 0.8 for 91% of positives; neither usable for only ≈ 1%.

**BIGGEST CHALLENGE:** Precision on look-alikes — sibling decoys and same-name/same-address entities whose distinguishing evidence (a unit number, a suffix word) is the same kind of perturbation the noise generator applies to true matches — compounded by (a) **France**, ≈ 15% of test entities, with no training data, 53% OOV name tokens, more name/address duplication and a new initials-only-name noise, and (b) India recall (native-script names, long shuffled addresses), which is ≈ 5 points below the US at every K.

**RECOMMENDED FIRST MODEL:** Normalisation (legal-form extraction, alias/ID/domain stripping, abbreviation + state maps, house-number feature) → per-country IDF retrieval (top-K 30–50 from address + name-token/prefix channels + exact tier) → ~35 name/address/context features → gradient-boosted trees trained on retriever candidate lists with grouped-by-S1 validation → one-to-one assignment + macro-F0.5-optimised threshold (≈ 0.6–0.7 in the diagnostic). Diagnostic version of this already gave macro-F0.5 ≈ 0.89 (US 0.94, India 0.85).

**RECOMMENDED BLOCKING:** `country` string equality, then a union of (a) exact token-sorted address/name equality, (b) IDF-weighted address-token retrieval, (c) IDF-weighted name-token + 4-char-prefix retrieval, ranked and cut at K ≈ 30–50 (recall ≈ 93–96% US, 87–91% India in the simple version; ≈ 35–90 M test pairs). Avoid first-token/first-N-char keys (2–14 k candidates/entity), state or city keys, house-number-only keys (lose 23–34%), and any rare-token rule with df ≤ 300 (≤ 65% recall).

**POTENTIAL DATA LEAKAGE RISKS:** *Not found:* ID values, ID length, row order, file position (Spearman ≈ 0, KS ≈ 0.005, ID-length agreement at chance), train/test ID overlap (0), train/test exact (name, address) overlap (0). *Found / to manage:* (1) record-level artefacts that reveal ‘matched’ status (empty address ⇒ 97.8%, `.com` names ⇒ ≈ 96%, shorter names, exact in-source duplicates ⇒ 100%) — legitimate at test time but they inflate validation if negatives are drawn only from matched records and may shift in France; (2) validation splits **must be by S1 entity** — names are template-generated (35% of test S1 names occur in train), so row-level splits leak; (3) random negatives are unrealistically easy (decoys are cleaner/longer than matched records) — train/validate on retrieved candidate lists; (4) the one-to-one partition property is a property of the *training* ground truth; it is unverifiable on test and should be used as a soft/verified assumption; (5) fitting dictionaries/IDF on test text is unsupervised but check the rules and document it.

**QUESTIONS WE SHOULD ANSWER BEFORE MODELING:** (1) Does test follow the same partition and ≈ 3.5 matches/entity, given 24% more S2/S3 per S1 (≈ 41% decoys inferred)? How much precision margin does that require? (2) What transfers to France — can leave-one-country-out and synthetic French noise be trusted as validation? Which French normalisation rules are needed? (3) Native-script names: token-translation table mined from training pairs vs multilingual model — what coverage/accuracy? (4) Can unit-number / suffix-word / legal-form-conflict features separate sibling decoys from true matches with mutated numbers, and what is the ceiling? (5) Where exactly is India’s recall lost at K=100 (90.7%)? (6) Best decision rule for macro-F0.5: global τ vs per-source/evidence τ vs per-entity expected-F0.5 subset selection; how much does the one-to-one rule buy? (7) Can S2↔S3 mutual similarity and in-source duplicates be used as consensus evidence? (8) K and candidate-file size (`candidate_pairs.tsv` ≈ 1 GB at K=50) — acceptable?

---
### Caveats and limitations of this reconnaissance
- Rates marked *(sample)* rest on 150k–500k pairs or 1,500–3,000 entities per country; headline percentages are stable to ≈ ±0.2–1 point, rare-category counts are not.
- The noise taxonomies are heuristic and priority-ordered; compound noise lands in the lowest-priority bucket. ‘Unrelated alias’ (4%) is defined by `token_sort_ratio` < 50, so it also contains some heavy-edit cases.
- The diagnostic model (§9, §11.5, §12) is a measuring instrument trained on tiny samples; its 0.89 is a floor-ish indication of what simple features achieve, not a forecast of leaderboard score, and it has no France information at all.
- Test-set statements about decoy share and France are **inferences from unsupervised proxies**; no test labels exist.
- Candidate-count experiments used same-country pools from the training split; test pools differ in size (US −38%, India +14%).
- Scripts: `analysis_scripts/` holds the main pipeline (`s00`–`s11`, `util.py`) and the report builders; caches (≈3 GB parquet) and intermediate JSON were written to a temporary scratch directory outside the project and are not part of the deliverable. A few small one-off checks were run interactively and are not saved (see the note at the top).
