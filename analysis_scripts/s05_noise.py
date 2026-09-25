import pandas as pd, numpy as np, json, re, random
from rapidfuzz import fuzz
from util import *
s1=load("train_source1"); pos=load("pos_pairs"); src={2:load("train_source2"),3:load("train_source3")}
rng=np.random.default_rng(3); S=pos.iloc[rng.choice(len(pos),500000,replace=False)].reset_index(drop=True)
rows=[]
for k in (2,3):
    q=S[S.src==k]; d=src[k]
    rows.append(pd.DataFrame({"src":k,"n1":s1.business_name.values[q.i1],"n2":d.business_name.values[q.j],
        "a1":s1.business_address.values[q.i1],"a2":d.business_address.values[q.j],"c":s1.country.values[q.i1]}))
D=pd.concat(rows,ignore_index=True)
INDIC=re.compile(r"[ऀ-෿]")
DBA=re.compile(r"\b(dba|d/b/a|f/k/a|fka|t/a|a/k/a|aka|formerly|trading as)\b|\bdba:|f\.k\.a",re.I)
HON=re.compile(r"^(mr|mrs|ms|shri|smt|sri|m/s|dr|messrs|late|kumari)\b\.?\s",re.I)
IDSUF=re.compile(r"\(ID:\s*\d+\)|#\s*\d{3,}\s*$")
JUNK=re.compile(r"^[^\wऀ-෿]+")
LEET=re.compile(r"[A-Za-z][0-9][A-Za-z]|^[0-9][A-Za-z]{3,}|[A-Za-z]{3,}[0-9]$|\bl[nN][cC]\b")
LEGAL=re.compile(r"\b(pvt|private|ltd|limited|llc|l\.l\.c\.?|llp|inc|incorporated|corp|corporation|co|company|pllc|pc|sarl|sas|sa|s\.a\.s|plc|lp|trust|tbk|gmbh)\b\.?",re.I)
n1b=norm_basic(D.n1); n2b=norm_basic(D.n2)
n1e=norm_ext(D.n1,ABBR_NAME); n2e=norm_ext(D.n2,ABBR_NAME)
t1=n1e.str.split().map(set); t2=n2e.str.split().map(set)
sim=np.array([fuzz.token_sort_ratio(a,b) for a,b in zip(n1e.values,n2e.values)])
D["sim"]=sim
def cat(i):
    a,b=D.n1.values[i],D.n2.values[i]
    if a==b: return "identical"
    if a.lower()==b.lower(): return "case_only"
    if n1b.values[i]==n2b.values[i]: return "whitespace/punct_only"
    if n1e.values[i]==n2e.values[i]: return "abbrev/diacritic/&_only"
    if sorted(n1e.values[i].split())==sorted(n2e.values[i].split()): return "word_order_only"
    if INDIC.search(b) and not INDIC.search(a): return "native_script_translit"
    if DBA.search(b): return "dba/fka/ta_alias"
    if t1.values[i]<t2.values[i]: return "extra_tokens_in_match(suffix/honorific/dup)"
    if t2.values[i]<t1.values[i]: return "dropped_tokens_in_match"
    if sim[i]>=80: return "typo/mixed_edits(sim>=80)"
    if sim[i]>=50: return "heavy_edits(50<=sim<80)"
    return "unrelated_or_alias(sim<50)"
D["cat"]=[cat(i) for i in range(len(D))]
order=D.cat.value_counts()
res={"name_taxonomy":{}}
for c,v in order.items():
    ex=D[D.cat==c].sample(min(6,v),random_state=1)
    res["name_taxonomy"][c]={"n":int(v),"pct":round(100*v/len(D),2),"examples":[[r.n1,r.n2,r.src] for r in ex.itertuples()]}
    for s in (2,3): res["name_taxonomy"][c][f"pct_S{s}"]=round(100*((D.cat==c)&(D.src==s)).sum()/(D.src==s).sum(),2)
    for cc in ("US","India"): res["name_taxonomy"][c][f"pct_{cc}"]=round(100*((D.cat==c)&(D.c==cc)).sum()/(D.c==cc).sum(),2)
# independent name flags on match side, and same flags on S1 side
def flags(s,label):
    f={"double_space":s.str.contains("  ",regex=False),"has_dba_fka_ta":s.str.contains(DBA),"honorific_prefix":s.str.contains(HON),
       "id_suffix":s.str.contains(IDSUF),"leading_junk_punct":s.str.contains(JUNK),"leet_or_digit_in_word":s.str.contains(LEET),
       "dot_com_domain":s.str.contains(r"\.com\b",case=False),"hash_or_at_prefix":s.str.match(r"^[#@]"),
       "parenthesized_or_bracketed":s.str.contains(r"[\(\[][^\)\]]+[\)\]]"),"ampersand":s.str.contains("&",regex=False),
       "word_and":s.str.contains(r"\band\b",case=False),"plus_sign":s.str.contains("+",regex=False),
       "all_upper":s.str.isupper(),"all_lower":s.str.islower(),"non_ascii_latin":(~s.map(str.isascii))&~s.str.contains(INDIC),
       "native_script":s.str.contains(INDIC),"legal_suffix_present":s.str.contains(LEGAL),
       "duplicate_adjacent_token":s.str.lower().str.contains(r"\b(\w{3,})\s+\1\b"),
       "single_token":s.str.split().str.len()==1}
    return {k:round(100*float(v.mean()),2) for k,v in f.items()}
res["name_flags_S1side"]=flags(D.n1,"s1"); res["name_flags_matchside_S2"]=flags(D.n2[D.src==2],"s2"); res["name_flags_matchside_S3"]=flags(D.n2[D.src==3],"s3")
# legal-suffix variants: which pairs differ in legal token
def legset(s): return frozenset(m.group(1).lower() for m in LEGAL.finditer(s))
ls1=D.n1.map(legset); ls2=D.n2.map(legset)
res["legal_suffix"]={"same_set":round(100*float((ls1==ls2).mean()),2),"match_has_none_S1_has":round(100*float(((ls2==frozenset())&(ls1!=frozenset())).mean()),2),
  "match_has_S1_none":round(100*float(((ls1==frozenset())&(ls2!=frozenset())).mean()),2)}
# concrete legal variant transitions
pairs=[]
for a,b in zip(ls1.values,ls2.values):
    if a!=b: pairs.append((",".join(sorted(a)) or "-",",".join(sorted(b)) or "-"))
res["legal_transitions_top"]=[[f"{a} -> {b}",c] for (a,b),c in pd.Series(pairs).value_counts().head(25).items()]
# specific example searches (name)
def find(mask,n=6):
    x=D[mask]; return [[r.n1,r.n2] for r in x.sample(min(n,len(x)),random_state=2).itertuples()]
ex={}
ex["Pvt<->Private"]=find((D.n1.str.contains(r"\bPvt\b",case=False)&D.n2.str.contains(r"\bPrivate\b",case=False))|(D.n1.str.contains(r"\bPrivate\b",case=False)&D.n2.str.contains(r"\bPvt\b",case=False)))
ex["Ltd<->Limited"]=find((D.n1.str.contains(r"\bLtd\b",case=False)&D.n2.str.contains(r"\bLimited\b",case=False))|(D.n1.str.contains(r"\bLimited\b",case=False)&D.n2.str.contains(r"\bLtd\b",case=False)))
ex["Corp<->Corporation"]=find((D.n1.str.contains(r"\bCorp\b",case=False)&D.n2.str.contains(r"\bCorporation\b",case=False))|(D.n1.str.contains(r"\bCorporation\b",case=False)&D.n2.str.contains(r"\bCorp\b",case=False)))
ex["Inc<->Incorporated"]=find((D.n1.str.contains(r"\bInc\b",case=False)&D.n2.str.contains(r"\bIncorporated\b",case=False))|(D.n1.str.contains(r"\bIncorporated\b",case=False)&D.n2.str.contains(r"\bInc\b",case=False)))
ex["Co<->Company"]=find((D.n1.str.contains(r"\bCo\b",case=False)&D.n2.str.contains(r"\bCompany\b",case=False))|(D.n1.str.contains(r"\bCompany\b",case=False)&D.n2.str.contains(r"\bCo\b",case=False)))
ex["&<->and"]=find((D.n1.str.contains("&",regex=False)&D.n2.str.contains(r"\band\b",case=False)&~D.n2.str.contains("&",regex=False))|(D.n1.str.contains(r"\band\b",case=False)&D.n2.str.contains("&",regex=False)&~D.n1.str.contains("&",regex=False)))
ex["LLC<->L.L.C."]=find(D.n2.str.contains(r"L\.L\.C",case=False)&D.n1.str.contains(r"\bLLC\b"))
ex["typo(sim 80-95, same token count)"]=find((D.sim>=80)&(D.sim<95)&(D.cat=="typo/mixed_edits(sim>=80)"))
ex["leet/digit substitution"]=find(D.n2.str.contains(LEET)&~D.n1.str.contains(LEET))
ex["diacritic injection (S1 ascii)"]=find(D.n1.map(str.isascii)&(~D.n2.map(str.isascii))&~D.n2.str.contains(INDIC))
ex["domain-form name"]=find(D.n2.str.contains(r"\.com\b",case=False)&~D.n1.str.contains(r"\.com\b",case=False))
ex["honorific added"]=find(D.n2.str.contains(HON)&~D.n1.str.contains(HON))
ex["(ID: n)/#n suffix"]=find(D.n2.str.contains(IDSUF))
ex["bracketed legal token"]=find(D.n2.str.contains(r"[\(\[](pvt|private|ltd|limited|llc|pc|inc|corp)[\)\]]",case=False))
ex["leading junk punctuation"]=find(D.n2.str.contains(JUNK))
ex["duplicated token"]=find(D.n2.str.lower().str.contains(r"\b(\w{3,})\s+\1\b")&~D.n1.str.lower().str.contains(r"\b(\w{3,})\s+\1\b"))
ex["transliteration to native script"]=find(D.n2.str.contains(INDIC)&~D.n1.str.contains(INDIC))
ex["DBA/FKA/TA"]=find(D.n2.str.contains(DBA))
ex["unrelated alias (sim<30, latin)"]=find((D.sim<30)&~D.n2.str.contains(INDIC))
ex["S1 itself has DBA"]=find(D.n1.str.contains(DBA))
res["name_examples"]=ex
res["dba_forms_matchside"]=D.n2[D.n2.str.contains(DBA)].str.extract(DBA)[0].str.lower().value_counts().head(10).to_dict()
res["alias_pattern_after_dba_is_s1_name_pct"]=None
# how often does DBA-form contain S1 name as substring
m=D[D.n2.str.contains(DBA)]; res["dba_contains_s1name_pct"]=round(100*float(np.mean([a.lower() in b.lower() for a,b in zip(m.n1,m.n2)])),2); res["dba_n"]=int(len(m))
# sim distribution
res["name_tokensort_sim_pctiles"]=D.sim.quantile([.05,.1,.25,.5,.75,.9]).to_dict()
# ---------------- ADDRESS
a1=D.a1; a2=D.a2; ne=(a2!="")
A1=norm_ext(a1,ABBR,True); A2=norm_ext(a2,ABBR,True)
T1=A1.str.split().map(set); T2=A2.str.split().map(set)
num1=a1.str.extract(r"(\d+)")[0]; num2=a2.str.extract(r"(\d+)")[0]
ab1=norm_basic(a1); ab2=norm_basic(a2)
def acat(i):
    a,b=a1.values[i],a2.values[i]
    if b=="": return "empty_address"
    if a==b: return "identical"
    if ab1.values[i]==ab2.values[i]: return "case/whitespace/punct_only"
    if A1.values[i]==A2.values[i]: return "abbrev/placeholder_only(Rd/Road, NULL dropped)"
    if sorted(A1.values[i].split())==sorted(A2.values[i].split()): return "component_reorder_only"
    if INDIC.search(b) and not INDIC.search(a): return "native_script_component"
    if T2.values[i]<T1.values[i]: return "match_missing_components(subset of S1)"
    if T1.values[i]<T2.values[i]: return "match_has_extra_components(PO Box/PMB/county...)"
    j=len(T1.values[i]&T2.values[i])/max(1,len(T1.values[i]|T2.values[i]))
    if j>=0.5: return "partial_overlap(jaccard>=0.5)"
    if j>0: return "weak_overlap(0<jaccard<0.5)"
    return "no_token_overlap"
D["acat"]=[acat(i) for i in range(len(D))]
res["addr_taxonomy"]={}
for c,v in D.acat.value_counts().items():
    ex=D[D.acat==c].sample(min(6,v),random_state=1)
    res["addr_taxonomy"][c]={"n":int(v),"pct":round(100*v/len(D),2),"examples":[[r.a1,r.a2,r.src] for r in ex.itertuples()]}
    for s in (2,3): res["addr_taxonomy"][c][f"pct_S{s}"]=round(100*((D.acat==c)&(D.src==s)).sum()/(D.src==s).sum(),2)
    for cc in ("US","India"): res["addr_taxonomy"][c][f"pct_{cc}"]=round(100*((D.acat==c)&(D.c==cc)).sum()/(D.c==cc).sum(),2)
def aflags(s):
    f={"NULL/N/A/<NULL>_placeholder":s.str.contains(r"(?:^|,\s*)(?:<NULL>|NULL|N/A)(?:,|$)",case=False),
       "PO_BOX":s.str.contains(r"\bP\.?\s?O\.?\s?Box\b",case=False),"PMB":s.str.contains(r"\bPMB\b",case=False),
       "unit_or_#":s.str.contains(r"(?:\bUnit\b|\bApt\b|\bSuite\b|#)",case=False),
       "leading_zero_number":s.str.contains(r"(?:^|[\s,#])0\d{2,}\b"),"leading_hash":s.str.contains(r"^#"),
       "double_space":s.str.contains("  ",regex=False),"all_upper":s.str.isupper(),
       "abbr_Rd/St/Ave/Dr/Ln/Ct/Cir/Blvd/Hwy":s.str.contains(r"\b(?:Rd|St|Ave|Dr|Ln|Ct|Cir|Blvd|Hwy|Pkwy|Ter|Trl|Cv)\b\.?",case=False),
       "full_Road/Street/Avenue/Drive/Lane/Court":s.str.contains(r"\b(?:Road|Street|Avenue|Drive|Lane|Court|Circle|Boulevard|Highway|Terrace|Trail|Cove)\b",case=False),
       "landmark(near/opp/behind/beside/next to)":s.str.contains(r"\b(?:near|opp\.?|opposite|behind|beside|next to|adj\.?|adjacent)\b",case=False),
       "native_script":s.str.contains(INDIC),"non_ascii_latin":(~s.map(str.isascii))&~s.str.contains(INDIC),
       "has_5digit_zip":s.str.contains(r"(?<!\d)\d{5}(?:-\d{4})?(?!\d)"),"has_6digit_PIN":s.str.contains(r"(?<!\d)[1-9]\d{5}(?!\d)"),
       "US_full_state_name_at_end":s.str.contains(r"(?:,\s)(?:Alabama|Alaska|Arizona|Arkansas|California|Colorado|Connecticut|Delaware|Florida|Georgia|Texas|Ohio|New York|Virginia|Washington|Illinois|Michigan|Pennsylvania|North Carolina|Tennessee|Kentucky)$"),
       "fraction_slash_1/2":s.str.contains(r"\d\s?1/2"),"suffix_letter_after_number(3707-D)":s.str.contains(r"^\d+-?[A-Za-z]\b")}
    return {k:round(100*float(v.mean()),2) for k,v in f.items()}
for lab,mask,col in [("S1side_US",(D.c=="US"),a1),("S1side_India",(D.c=="India"),a1),("S2side_US",(D.c=="US")&(D.src==2),a2),("S2side_India",(D.c=="India")&(D.src==2),a2),("S3side_US",(D.c=="US")&(D.src==3),a2),("S3side_India",(D.c=="India")&(D.src==3),a2)]:
    res.setdefault("addr_flags",{})[lab]=aflags(col[mask&(col!="")])
def afind(mask,n=6):
    x=D[mask&(D.a2!="")]; return [[r.a1,r.a2,r.src] for r in x.sample(min(n,len(x)),random_state=2).itertuples()]
e={}
e["Rd<->Road / St<->Street (abbrev vs full)"]=afind((a1.str.contains(r"\b(Road|Street|Avenue|Drive|Lane)\b"))&(a2.str.contains(r"\b(Rd|St|Ave|Dr|Ln)\b",case=False))&(D.c=="US"))
e["missing city/state"]=afind((D.c=="US")&(~a2.str.contains(r",\s*[A-Z]{2}$"))&(~a2.str.contains(r"[A-Za-z]+,\s*(?:[A-Z][a-z]+\s?)+$"))&(D.acat.str.startswith("match_missing")))
e["missing state (US) - only street+city"]=afind((D.c=="US")&(D.acat.str.startswith("match_missing"))&(a2.str.count(",")<=1))
e["reordered components"]=afind(D.acat=="component_reorder_only")
e["landmark (India)"]=afind(a1.str.contains(r"\b(?:Near|Opp|Behind|Beside|Opposite)\b",case=False)&(D.c=="India"))
e["PO Box/PMB inserted in match"]=afind(a2.str.contains(r"PO Box|PMB",case=False)&~a1.str.contains(r"PO Box|PMB",case=False))
e["NULL/N/A placeholder"]=afind(a2.str.contains(r"(?:<NULL>|\bNULL\b|N/A)"))
e["leading-zero house number"]=afind(a2.str.contains(r"(?:^|[\s,#])0\d{2,}\b")&~a1.str.contains(r"(?:^|[\s,#])0\d{2,}\b"))
e["house number differs (first number)"]=afind((num1!=num2)&num1.notna()&num2.notna()&(D.c=="US"))
e["city misspelled / substituted"]=afind((D.c=="US")&(D.acat=="partial_overlap(jaccard>=0.5)"))
e["native-script state/city (India)"]=afind(a2.str.contains(INDIC)&(D.c=="India"))
e["US full state vs abbreviation"]=afind((D.c=="US")&(D.src==3)&a2.str.contains(r"(?:Texas|Ohio|California|Florida|New York|Georgia|Illinois)$"))
e["India state code vs full (GJ/Gujarat)"]=afind((D.c=="India")&(D.src==3)&a2.str.contains(r",\s[A-Z]{2}$"))
e["India transliteration variants (spelling)"]=afind((D.c=="India")&(D.acat=="partial_overlap(jaccard>=0.5)"))
e["no token overlap"]=afind(D.acat=="no_token_overlap")
res["addr_examples"]=e
# house number equality
both=(num1.notna()&num2.notna())
res["house_number"]={"both_have_number_pct":round(100*float(both.mean()),2),"first_number_equal_pct_of_both":round(100*float((num1[both]==num2[both]).mean()),2),
  "equal_after_strip_leading_zeros_pct_of_both":round(100*float((num1[both].str.lstrip("0")==num2[both].str.lstrip("0")).mean()),2)}
for cc in ("US","India"):
    m=both&(D.c==cc); res["house_number"][f"first_number_equal_{cc}"]=round(100*float((num1[m].str.lstrip("0")==num2[m].str.lstrip("0")).mean()),2)
# token jaccard address
J=np.array([len(x&y)/max(1,len(x|y)) if b else np.nan for x,y,b in zip(T1.values,T2.values,ne.values)])
res["addr_token_jaccard_pctiles_nonempty"]=pd.Series(J).quantile([.05,.1,.25,.5,.75,.9]).to_dict()
res["addr_jaccard_zero_pct_nonempty"]=round(100*float(np.nanmean(J[~np.isnan(J)]==0)),3)
json.dump(res,open("out/noise.json","w"),indent=1,ensure_ascii=False,default=str)
D[["src","c","cat","acat","sim"]].to_parquet("out/noise_sample_cats.parquet")
print("done")
