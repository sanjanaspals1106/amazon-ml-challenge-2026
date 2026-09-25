import pandas as pd, numpy as np, json, re, time
from collections import Counter
from util import sort_tokens
t0=time.time(); R={}
def L(n): return pd.read_parquet(f"cache/norm_{n}.parquet")
# ---- (c) singleton vs non-singleton at S1 level
raw=pd.read_parquet("cache/train_source1.parquet"); N=L("train_source1"); gc=pd.read_parquet("cache/gt_counts.parquet").set_index("sid").n
n=gc.reindex(raw.entity_id.values).values; single=n==0
D=pd.DataFrame({"country":raw.country.values,"single":single,"n":n,"nn":N.nn.values,"na":N.na.values,"name_len":raw.business_name.str.len().values,"addr_len":raw.business_address.str.len().values})
D["ntok"]=D.nn.str.split().str.len(); D["atok"]=D.na.str.split().str.len()
D["name_mult"]=D.groupby("nn").nn.transform("size"); D["addr_mult"]=D.groupby("na").na.transform("size")
tokdf=Counter(); 
for x in D.nn.sample(600000,random_state=1).values: tokdf.update(set(x.split()))
D["name_min_tokdf"]=D.nn.map(lambda s:min([tokdf.get(t,0) for t in s.split()] or [0]))
D["legal"]=D.nn.str.contains(r"\b(?:private limited|limited|ltd|llc|llp|inc|incorporated|corp|corporation|company|pllc|pc)\b")
R["singleton_rate_by_country"]=D.groupby("country").single.mean().round(4).to_dict(); R["singleton_rate"]=round(float(D.single.mean()),4)
g=D.groupby("single")[["name_len","addr_len","ntok","atok","name_mult","addr_mult","name_min_tokdf","legal"]].mean().round(3)
R["singleton_vs_non_S1_attrs"]={str(k):v for k,v in g.to_dict().items()}
for c in ("US","India"):
    R[f"singleton_vs_non_{c}"]={str(k):v for k,v in D[D.country==c].groupby("single")[["name_len","addr_len","ntok","name_mult","addr_mult","name_min_tokdf","legal"]].mean().round(3).to_dict().items()}
D["nm_bin"]=pd.cut(D.name_mult,[0,1,2,5,20,1000]); R["singleton_rate_by_name_multiplicity"]=D.groupby("nm_bin",observed=True).single.mean().round(4).rename(index=str).to_dict()
D["am_bin"]=pd.cut(D.addr_mult,[0,1,2,1000]); R["singleton_rate_by_addr_multiplicity"]=D.groupby("am_bin",observed=True).single.mean().round(4).rename(index=str).to_dict()
D["nl_bin"]=pd.cut(D.name_len,[0,10,15,20,30,40,200]); R["singleton_rate_by_name_len"]=D.groupby("nl_bin",observed=True).single.mean().round(4).rename(index=str).to_dict()
R["singleton_rate_by_n_matches_context"]=D.groupby("legal").single.mean().round(4).rename(index=str).to_dict()
R["singleton_examples"]=raw[single].sample(12,random_state=4)[["business_name","business_address","country"]].values.tolist()
print("singletons",round(time.time()-t0),flush=True)
# ---- (a) France cities / regions
te=pd.read_parquet("cache/test_source1.parquet"); fr=te[te.country=="France"]
parts=fr.business_address.str.split(", ")
R["france_S1_last_component_top"]=parts.str[-1].value_counts().head(12).to_dict()
cities=fr.business_address.str.extract(r"(?:,\s)([^,]+)(?:,\s[^,]+)?$")[0]
R["france_S1_distinct_last_comp"]=int(parts.str[-1].nunique()); R["france_S1_distinct_second_last_comp"]=int(parts.str[-2].nunique())
R["france_S1_second_last_top"]=parts.str[-2].value_counts().head(15).to_dict()
R["france_S1_name_first_token_top"]=fr.business_name.str.split().str[0].value_counts().head(15).to_dict()
# ---- (b) test matched-fraction proxy: share of S2/S3 records whose sorted-token address appears in S1 (same split)
def keys(n):
    d=L(n); m=d.na!=""; return pd.DataFrame({"c":d.country[m].values,"k":sort_tokens(d.na[m]).values})
out={}
for sp_ in ("train","test"):
    k1=keys(f"{sp_}_source1"); s1set=set(zip(k1.c,k1.k))
    for s in (2,3):
        k=keys(f"{sp_}_source{s}"); k["h"]=[(a,b) in s1set for a,b in zip(k.c,k.k)]
        out[f"{sp_}_S{s}"]=k.groupby("c").h.mean().round(4).to_dict()
    print(sp_,round(time.time()-t0),flush=True)
R["proxy_share_of_S2S3_records_with_addr_in_S1"]=out
# ---- pair-space size
cnt=json.load(open("out/traintest.json"))["country_counts"]
for sp_ in ("train","test"):
    tot=0; percountry={}
    for c in cnt[f"{sp_}_S1"]:
        p=cnt[f"{sp_}_S1"][c]*(cnt[f"{sp_}_S2"][c]+cnt[f"{sp_}_S3"][c]); percountry[c]=p; tot+=p
    R[f"cross_pairs_same_country_{sp_}"]={"total":tot,"by_country":percountry}
    R[f"cross_pairs_all_{sp_}"]=sum(cnt[f"{sp_}_S1"].values())*sum(cnt[f"{sp_}_S2"][c]+cnt[f"{sp_}_S3"][c] for c in cnt[f"{sp_}_S2"])
json.dump(R,open("out/extra.json","w"),indent=1,ensure_ascii=False,default=str); print(json.dumps(R,indent=1,ensure_ascii=False,default=str)[:6000])
