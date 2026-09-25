import pandas as pd, numpy as np, json, time
from util import *
s1=load("train_source1"); pos=load("pos_pairs")
src={2:load("train_source2"),3:load("train_source3")}
n1=s1.business_name.values; a1=s1.business_address.values; c1=s1.country.values
out=[]; CH=400000; t=time.time()
for k in (2,3):
    d=src[k]; p=pos[pos.src==k].reset_index(drop=True)
    for st in range(0,len(p),CH):
        q=p.iloc[st:st+CH]
        A=pd.Series(n1[q.i1.values]); B=pd.Series(d.business_name.values[q.j.values])
        AA=pd.Series(a1[q.i1.values]); BB=pd.Series(d.business_address.values[q.j.values])
        r=pd.DataFrame({"i1":q.i1.values,"j":q.j.values,"src":k})
        r["name_raw"]=(A==B).values
        r["name_lower"]=(A.str.lower()==B.str.lower()).values
        nb1,nb2=norm_basic(A),norm_basic(B); r["name_basic"]=(nb1==nb2).values
        ne1,ne2=norm_ext(A,ABBR_NAME),norm_ext(B,ABBR_NAME); r["name_ext"]=(ne1==ne2).values
        r["name_sorted"]=(sort_tokens(ne1)==sort_tokens(ne2)).values
        r["addr_empty"]=(BB=="").values
        r["addr_raw"]=(AA==BB).values
        ab1,ab2=norm_basic(AA),norm_basic(BB); r["addr_basic"]=((ab1==ab2)&(BB!="")).values
        ae1,ae2=norm_ext(AA,ABBR,True),norm_ext(BB,ABBR,True); r["addr_ext"]=((ae1==ae2)&(BB!="")).values
        r["addr_sorted"]=((sort_tokens(ae1)==sort_tokens(ae2))&(BB!="")).values
        r["country_same"]=(c1[q.i1.values]==d.country.values[q.j.values])
        r["c1"]=c1[q.i1.values]
        out.append(r)
    print(k,round(time.time()-t),flush=True)
R=pd.concat(out,ignore_index=True); R.to_parquet(f"{C}/pair_exact_flags.parquet")
print(R.shape)
