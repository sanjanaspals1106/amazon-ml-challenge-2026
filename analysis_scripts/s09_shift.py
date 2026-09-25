import pandas as pd, numpy as np, json, re, time
from collections import Counter
from scipy.stats import ks_2samp, spearmanr
from util import sort_tokens
t0=time.time(); rng=np.random.default_rng(9); R={}
def L(n): return pd.read_parquet(f"cache/norm_{n}.parquet")
raw={n:pd.read_parquet(f"cache/{n}.parquet") for n in ["train_source1","test_source1"]}
NN={n:L(n) for n in ["train_source1","test_source1"]}
# ---------- 1. per-country counts & ratios
cnt={}
for sp_ in ("train","test"):
    for s in (1,2,3):
        c=pd.read_parquet(f"cache/{sp_}_source{s}.parquet",columns=["country"]).country.value_counts()
        cnt[f"{sp_}_S{s}"]=c.to_dict()
R["country_counts"]=cnt
# ---------- 2. S1 duplication structure per country, train vs test
def dupstats(df,nn):
    out={}
    for c,g in df.groupby("country").groups.items():
        d=df.loc[g]; n=nn.loc[g]
        vn=n.nn.value_counts(); va=d.business_address.value_counts()
        out[c]={"rows":len(d),"unique_norm_names":int(len(vn)),"rows_sharing_name_pct":round(100*float(vn[vn>1].sum()/len(d)),2),"max_name_multiplicity":int(vn.max()),
                "unique_addresses":int(len(va)),"rows_sharing_address_pct":round(100*float(va[va>1].sum()/len(d)),2),"max_addr_multiplicity":int(va.max()),
                "name_len_mean":round(float(d.business_name.str.len().mean()),2),"addr_len_mean":round(float(d.business_address.str.len().mean()),2),
                "addr_tokens_mean":round(float(d.business_address.str.split().str.len().mean()),2)}
    return out
R["S1_dup_structure"]={"train":dupstats(raw["train_source1"],NN["train_source1"]),"test":dupstats(raw["test_source1"],NN["test_source1"])}
print("dup",round(time.time()-t0),flush=True)
# ---------- 3. legal suffix mix in S1 names by country
LEG=re.compile(r"\b(private limited|pvt ltd|pvt|private|limited|ltd|llc|llp|inc|incorporated|corp|corporation|co|company|pllc|pc|sarl|sas|sci|eurl|sa|earl|gmbh|trust|club|association|amicale)\b")
leg={}
for sp_ in ("train","test"):
    d=NN[f"{sp_}_source1"]; s=d.sample(300000,random_state=1)
    for c in s.country.unique():
        t=s[s.country==c].nn.str.findall(LEG.pattern).explode().value_counts(normalize=True).head(12).round(4).to_dict(); leg[f"{sp_}_{c}"]=t
R["S1_legal_token_share"]=leg
# ---------- 4. token vocabulary shift / OOV (test vs train, same country) 
def toks(ser,n=300000):
    c=Counter()
    for x in ser.sample(min(n,len(ser)),random_state=2).values: c.update(x.split())
    return c
oov={}
for view,col in (("name","nn"),("addr","na")):
    for c in ("US","India"):
        trv=Counter(); 
        for s in (1,2,3): 
            d=L(f"train_source{s}"); trv.update(toks(d[d.country==c][col],200000))
        # baseline: train-holdout vs train (S1 second half)
        d1=L("train_source1"); d1=d1[d1.country==c][col]; a=d1.sample(150000,random_state=5); 
        ref=Counter(); 
        for s in (2,3):
            d=L(f"train_source{s}"); ref.update(toks(d[d.country==c][col],200000))
        tc=toks(L("test_source1").query("country==@c")[col],150000); bc=toks(a,150000)
        def rate(cn,vocab): tot=sum(cn.values()); return round(100*sum(v for k,v in cn.items() if k not in vocab)/tot,3), round(100*sum(1 for k in cn if k not in vocab)/len(cn),2)
        oov[f"{view}_{c}"]={"test_S1_tokens_OOV_vs_trainS2S3_pct(token-weighted, type)":rate(tc,ref),"train_S1_tokens_OOV_vs_trainS2S3_pct(baseline)":rate(bc,ref)}
    fr=toks(L("test_source1").query("country=='France'")[col],150000)
    allv=Counter()
    for s in (1,2,3): 
        d=L(f"train_source{s}"); allv.update(toks(d[col],150000))
    tot=sum(fr.values()); oov[f"{view}_France_vs_all_train"]={"token_weighted_OOV_pct":round(100*sum(v for k,v in fr.items() if k not in allv)/tot,2),"top_tokens":[k for k,_ in fr.most_common(40)]}
R["oov"]=oov; print("oov",round(time.time()-t0),flush=True)
# ---------- 5. France structure
te={s:pd.read_parquet(f"cache/test_source{s}.parquet") for s in (1,2,3)}
fr={}
for s in (1,2,3):
    d=te[s][te[s].country=="France"]
    fr[f"S{s}"]={"rows":len(d),"empty_addr_pct":round(100*float((d.business_address=="").mean()),2),"name_nonascii_pct":round(100*float((~d.business_name.map(str.isascii)).mean()),2),
        "unique_names":int(d.business_name.nunique()),"upper_name_pct":round(100*float(d.business_name.str.isupper().mean()),2),"upper_addr_pct":round(100*float(d.business_address[d.business_address!=""].str.isupper().mean()),2),
        "addr_has_5digit_postcode_pct":round(100*float(d.business_address.str.contains(r"(?<!\d)\d{5}(?!\d)").mean()),2),
        "street_type_counts":d.business_address.str.extract(r"(?i)\b(rue|r\.|av|avenue|bd|boulevard|chemin|place|pl|impasse|route|quai|all[ée]e|cours)\b")[0].str.lower().value_counts().head(10).to_dict(),
        "legal_forms":d.business_name.str.extract(r"\b(SARL|SAS|SASU|SCI|EURL|EARL|SA|GAEC|SNC|SCOP|SELARL|S\.A\.S|S\.A\.R\.L)\b",flags=re.I)[0].str.upper().value_counts().head(10).to_dict(),
        "sample_names":d.business_name.sample(8,random_state=1).tolist(),"sample_addresses":d.business_address[d.business_address!=""].sample(8,random_state=1).tolist()}
R["france"]=fr
# ---------- 6. unsupervised match-density proxy: does S1 entity have any S2/S3 record with identical (sorted-token) address? by country, train vs test
def akeys(n): 
    d=L(n); d=d[d.na!=""]; return d.country.values, sort_tokens(d.na).values
proxy={}
for sp_ in ("train","test"):
    c1,k1=akeys(f"{sp_}_source1"); s1df=pd.DataFrame({"c":c1,"k":k1})
    ks=set()
    for s in (2,3):
        c,k=akeys(f"{sp_}_source{s}"); ks.update(zip(c,k))
    s1df["hit"]=[(c,k) in ks for c,k in zip(s1df.c,s1df.k)]
    proxy[sp_]=s1df.groupby("c").hit.mean().round(4).to_dict()
R["proxy_S1_has_exact_addr_in_S2S3"]=proxy; print("proxy",round(time.time()-t0),flush=True)
# ---------- 7. leakage / record-level priors on train S2, S3: matched vs unmatched
pos=pd.read_parquet("cache/pos_pairs.parquet"); rp={}
for s in (2,3):
    d=pd.read_parquet(f"cache/train_source{s}.parquet"); m=np.zeros(len(d),bool); m[pos[pos.src==s].j.values]=True
    d["matched"]=m; idn=d.entity_id.str.extract(r"-(\d+)")[0].astype("int64")
    f=pd.DataFrame({"matched":m,"country_US":(d.country=="US").values,"addr_empty":(d.business_address=="").values,"name_len":d.business_name.str.len().values,"addr_len":d.business_address.str.len().values,
        "name_upper":d.business_name.str.isupper().values,"name_native":d.business_name.str.contains(r"[ऀ-෿]").values,"dotcom":d.business_name.str.contains(r"\.com",case=False).values,
        "name_ntokens":d.business_name.str.split().str.len().values,"double_space":d.business_name.str.contains("  ",regex=False).values,"id_len":d.entity_id.str.len().values,"id_num":idn.values,
        "row_pos_frac":np.arange(len(d))/len(d)})
    g=f.groupby("matched").mean().round(4); rp[f"S{s}"]=g.to_dict()
    rp[f"S{s}_KS"]={c:round(float(ks_2samp(f[c][f.matched].sample(100000,random_state=1),f[c][~f.matched].sample(100000,random_state=1)).statistic),4) for c in ("id_num","row_pos_frac","name_len","addr_len")}
    # matched fraction by country
    rp[f"S{s}_matched_frac_by_country"]=d.groupby("country").matched.mean().round(4).to_dict()
R["matched_vs_unmatched_priors"]=rp
# id/pos correlations
s1=pd.read_parquet("cache/train_source1.parquet",columns=["entity_id"]); idn1=s1.entity_id.str.extract(r"-(\d+)")[0].astype("int64").values
samp=pos.sample(300000,random_state=3)
d2=pd.read_parquet("cache/train_source2.parquet",columns=["entity_id"]); d3=pd.read_parquet("cache/train_source3.parquet",columns=["entity_id"])
idj=np.where(samp.src.values==2,d2.entity_id.str.extract(r"-(\d+)")[0].astype("int64").values[np.where(samp.src.values==2,samp.j.values,0)],d3.entity_id.str.extract(r"-(\d+)")[0].astype("int64").values[np.where(samp.src.values==3,samp.j.values,0)])
R["id_leak"]={"spearman_S1id_vs_matchid":round(float(spearmanr(idn1[samp.i1.values],idj).correlation),4),"spearman_S1rowpos_vs_matchrowpos":round(float(spearmanr(samp.i1.values/len(s1),samp.j.values/np.where(samp.src.values==2,len(d2),len(d3))).correlation),4),
   "id_digit_len_equal_pct":round(100*float((pd.Series(idn1[samp.i1.values]).astype(str).str.len().values==pd.Series(idj).astype(str).str.len().values).mean()),2),
   "id_digit_len_equal_expected_if_random_pct":round(100*float((pd.Series(idn1[rng.integers(0,len(s1),300000)]).astype(str).str.len().values==pd.Series(idj).astype(str).str.len().values).mean()),2)}
# train/test ID overlap
ids={}
for sp_ in ("train","test"):
    for s in (1,2,3): ids[f"{sp_}{s}"]=set(pd.read_parquet(f"cache/{sp_}_source{s}.parquet",columns=["entity_id"]).entity_id.values)
R["id_overlap_train_test"]={f"S{s}":len(ids[f"train{s}"]&ids[f"test{s}"]) for s in (1,2,3)}
# exact content overlap train/test S1 (leak between splits)
a=set(zip(NN["train_source1"].nn,NN["train_source1"].na)); b=list(zip(NN["test_source1"].nn,NN["test_source1"].na))
R["S1_test_rows_whose_norm_name+addr_appears_in_train_S1"]=round(100*float(np.mean([x in a for x in b])),4)
an=set(NN["train_source1"].nn.values); R["S1_test_norm_name_seen_in_train_S1_pct"]=round(100*float(NN["test_source1"].nn.isin(an).mean()),2)
R["S1_test_norm_name_seen_in_train_S1_pct_by_country"]=NN["test_source1"].assign(h=NN["test_source1"].nn.isin(an)).groupby("country").h.mean().round(4).to_dict()
aa=set(NN["train_source1"].na.values); R["S1_test_norm_addr_seen_in_train_S1_pct_by_country"]=NN["test_source1"].assign(h=NN["test_source1"].na.isin(aa)).groupby("country").h.mean().round(4).to_dict()
print("leak",round(time.time()-t0),flush=True)
# ---------- 8. decoy (unmatched S2/S3) structure: do they cluster with each other (entities absent from S1)?
dec={}
N2=L("train_source2"); N3=L("train_source3")
m2=np.zeros(len(N2),bool); m2[pos[pos.src==2].j.values]=True; m3=np.zeros(len(N3),bool); m3[pos[pos.src==3].j.values]=True
k2=sort_tokens(N2.na).values; k3=sort_tokens(N3.na).values
k1=set(sort_tokens(NN["train_source1"].na).values)
def share(keys,mask,ref): 
    k=pd.Series(keys[mask]); k=k[k!=""].sample(150000,random_state=1); return round(100*float(k.isin(ref).mean()),2)
set_k3_un=set(k3[~m3]); set_k3_m=set(k3[m3]); set_k2_un=set(k2[~m2])
dec["S2_matched_addr_in_S1_pct"]=share(k2,m2,k1); dec["S2_unmatched_addr_in_S1_pct"]=share(k2,~m2,k1)
dec["S2_unmatched_addr_in_unmatched_S3_pct"]=share(k2,~m2,set_k3_un); dec["S2_unmatched_addr_in_matched_S3_pct"]=share(k2,~m2,set_k3_m)
dec["S2_matched_addr_in_unmatched_S3_pct"]=share(k2,m2,set_k3_un)
dec["S3_unmatched_addr_in_unmatched_S2_pct"]=share(k3,~m3,set_k2_un)
nn1=set(NN["train_source1"].nn.values); n2n=N2.nn.values
dec["S2_matched_normname_in_S1_pct"]=round(100*float(pd.Series(n2n[m2]).sample(150000,random_state=1).isin(nn1).mean()),2); dec["S2_unmatched_normname_in_S1_pct"]=round(100*float(pd.Series(n2n[~m2]).sample(150000,random_state=1).isin(nn1).mean()),2)
# unmatched: examples
d2raw=pd.read_parquet("cache/train_source2.parquet"); dec["S2_unmatched_examples"]=d2raw[~m2].sample(12,random_state=3)[["business_name","business_address","country"]].values.tolist()
dec["S2_unmatched_country"]=d2raw[~m2].country.value_counts(normalize=True).round(4).to_dict(); dec["S2_matched_country"]=d2raw[m2].country.value_counts(normalize=True).round(4).to_dict()
R["decoys"]=dec
json.dump(R,open("out/traintest.json","w"),indent=1,ensure_ascii=False,default=str); print("done",round(time.time()-t0))
