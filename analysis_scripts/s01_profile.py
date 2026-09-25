import pandas as pd, numpy as np, json, re, unicodedata, os, time
from collections import Counter
C="cache"
R={}
def lenstats(s):
    l=s.str.len()
    q=l.quantile([.01,.05,.25,.5,.75,.95,.99,.999]).to_dict()
    return dict(min=int(l.min()),max=int(l.max()),mean=round(float(l.mean()),2),std=round(float(l.std()),2),
                **{f"p{k*100:g}":float(v) for k,v in q.items()})
def script_of(ch):
    if ch.isascii(): return "ascii"
    try: n=unicodedata.name(ch)
    except: return "unknown"
    return n.split()[0]
def profile(name):
    t=time.time()
    df=pd.read_parquet(f"{C}/{name}.parquet")
    r={"rows":len(df),"cols":list(df.columns)}
    r["file_bytes"]=os.path.getsize(f"/Users/sanjana1106/Desktop/MLChallenge/student_resource/dataset/{name.split('_')[0]}/{name}.tsv")
    for c in df.columns:
        s=df[c]
        empty=(s=="").sum(); ws=(s.str.strip()=="").sum()
        r.setdefault("cols_detail",{})[c]={"empty":int(empty),"blank_or_ws":int(ws),"missing_pct":round(100*ws/len(df),3),
            "examples":s[s.str.strip()!=""].sample(5,random_state=1).tolist() if ws<len(df) else [],
            "unique":int(s.nunique())}
    if "entity_id" in df:
        ids=df.entity_id
        pre=name.split("_")[1].replace("source","S")
        r["id_pattern_ok"]=int(ids.str.fullmatch(rf"{pre}-\d+").sum())
        r["id_pattern_bad"]=int((~ids.str.fullmatch(rf"{pre}-\d+")).sum())
        r["id_bad_examples"]=ids[~ids.str.fullmatch(rf"{pre}-\d+")].head(5).tolist()
        r["unique_ids"]=int(ids.nunique()); r["dup_id_rows"]=int(ids.duplicated(keep=False).sum())
        r["id_numeric_min"]=int(ids.str.extract(r"-(\d+)")[0].astype("int64").min()); r["id_numeric_max"]=int(ids.str.extract(r"-(\d+)")[0].astype("int64").max())
        n=df.business_name; a=df.business_address
        r["unique_names"]=int(n.nunique()); 
        vc=n.value_counts(); r["names_appearing_gt1"]=int((vc>1).sum()); r["rows_with_dup_name"]=int(vc[vc>1].sum()); r["top_names"]=vc.head(15).to_dict()
        nl=n.str.lower().str.strip(); r["unique_names_lower"]=int(nl.nunique())
        va=a.value_counts(); r["unique_addresses"]=int(a.nunique()); r["addresses_appearing_gt1"]=int((va>1).sum()); r["rows_with_dup_address"]=int(va[va>1].sum()); r["top_addresses"]=va.head(15).to_dict()
        pair=(n+"|"+a).value_counts(); r["dup_name_addr_pairs"]=int((pair>1).sum()); r["rows_in_dup_name_addr"]=int(pair[pair>1].sum())
        r["country_dist"]=df.country.value_counts().to_dict()
        r["name_len"]=lenstats(n); r["addr_len"]=lenstats(a); r["addr_len_nonempty"]=lenstats(a[a!=""]) if (a!="").any() else None
        r["name_tokens"]=round(float(n.str.split().str.len().mean()),2); r["addr_tokens"]=round(float(a.str.split().str.len().mean()),2)
        # data quality
        dq={}
        for c in ["business_name","business_address"]:
            s=df[c]
            dq[c]={"leading_trailing_ws":int((s!=s.str.strip()).sum()),"double_space":int(s.str.contains("  ",regex=False).sum()),
                   "has_control":int(s.str.contains(r"[\x00-\x08\x0b-\x1f\x7f]",regex=True).sum()),
                   "replacement_char":int(s.str.contains("�",regex=False).sum()),
                   "mojibake_pat":int(s.str.contains(r"Ã.|â€|Â[^\w]|Ð.|à¤|à¥",regex=True).sum()),
                   "html_entity":int(s.str.contains(r"&(?:amp|lt|gt|quot|#\d+);",regex=True).sum()),
                   "literal_null_like":int(s.str.lower().str.strip().isin(["nan","null","none","n/a","na","-","--","unknown","nil","."]).sum()),
                   "non_ascii_rows":int((~s.map(str.isascii)).sum()),
                   "len_gt_150":int((s.str.len()>150).sum()),"len_gt_300":int((s.str.len()>300).sum()),
                   "only_punct_digits":int(s.str.fullmatch(r"[\W\d_]+").sum()),
                   "len_le_2_nonempty":int(((s.str.len()<=2)&(s!="")).sum())}
            dq[c]["long_examples"]=s[s.str.len()>200].head(3).str[:300].tolist()
            dq[c]["short_examples"]=s[(s.str.len()<=2)&(s!="")].head(8).tolist()
        r["dq"]=dq
        # script breakdown on non-ascii rows sample
        sub=df[~df.business_name.map(str.isascii)].business_name
        smp=sub.sample(min(20000,len(sub)),random_state=0) if len(sub) else sub
        cnt=Counter()
        for x in smp:
            for ch in set(x):
                if not ch.isascii() and ch.isalpha(): cnt[script_of(ch)]+=1
        r["name_nonascii_scripts_sample"]=dict(cnt.most_common(12)); r["name_nonascii_sample_n"]=len(smp)
        r["name_nonascii_by_country"]=df[~df.business_name.map(str.isascii)].country.value_counts().to_dict()
        sub=df[~df.business_address.map(str.isascii)].business_address
        smp=sub.sample(min(20000,len(sub)),random_state=0) if len(sub) else sub
        cnt=Counter()
        for x in smp:
            for ch in set(x):
                if not ch.isascii() and ch.isalpha(): cnt[script_of(ch)]+=1
        r["addr_nonascii_scripts_sample"]=dict(cnt.most_common(12))
        r["addr_nonascii_by_country"]=df[~df.business_address.map(str.isascii)].country.value_counts().to_dict()
        # missing by country
        r["missing_addr_by_country"]=df[df.business_address.str.strip()==""].country.value_counts().to_dict()
        r["missing_name_by_country"]=df[df.business_name.str.strip()==""].country.value_counts().to_dict()
        r["country_repr"]=[repr(x) for x in df.country.unique()[:20]]
    print(name,round(time.time()-t,1),flush=True)
    return r
for f in ["train_source1","train_source2","train_source3","test_source1","test_source2","test_source3","train_ground_truth"]:
    R[f]=profile(f)
json.dump(R,open("profile.json","w"),indent=1,default=str,ensure_ascii=False)
