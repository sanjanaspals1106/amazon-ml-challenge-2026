import pandas as pd, numpy as np, scipy.sparse as sp, json, time, re
from sklearn.feature_extraction.text import CountVectorizer
t0=time.time(); rng=np.random.default_rng(5)
N1=pd.read_parquet("cache/norm_train_source1.parquet"); N2=pd.read_parquet("cache/norm_train_source2.parquet"); N3=pd.read_parquet("cache/norm_train_source3.parquet")
pos=pd.read_parquet("cache/pos_pairs.parquet")
CORP=pd.concat([N2.assign(src=2),N3.assign(src=3)],ignore_index=True); n2=len(N2)
pos["cj"]=np.where(pos.src==2,pos.j,pos.j+n2)
def p4(s): return s.map(lambda x:" ".join(t[:4] for t in x.split() if len(t)>=3))
res={}
M=1500; NP=120000
for country in ("US","India"):
    print(country,flush=True); R={}
    cidx=np.where(CORP.country.values==country)[0]; C=CORP.iloc[cidx]; csrc=C.src.values
    remap=np.full(len(CORP),-1); remap[cidx]=np.arange(len(cidx))
    s1i=np.where(N1.country.values==country)[0]
    vecs={}; Xc={}; 
    views={"name":C.nn,"name_p4":p4(C.nn),"addr":C.na}
    for v,ser in views.items():
        cv=CountVectorizer(token_pattern=r"\S+",binary=True,dtype=np.int8,lowercase=False); Xc[v]=cv.fit_transform(ser.values).tocsr(); vecs[v]=cv
        print(" vec",v,Xc[v].shape,Xc[v].nnz,round(time.time()-t0),flush=True)
    df={v:np.asarray(Xc[v].sum(0)).ravel() for v in Xc}
    R["corpus_docs"]=len(cidx); R["vocab"]={v:len(vecs[v].vocabulary_) for v in vecs}
    R["df_pctiles_of_tokens"]={v:{str(q):float(np.quantile(df[v],q)) for q in (.5,.9,.99,.999)} for v in df}
    # ---- positive pair recall via min shared-token df
    pp=pos[pos.i1.isin(s1i)]; pp=pp.iloc[rng.choice(len(pp),min(NP,len(pp)),replace=False)]
    S1v={"name":N1.nn.values,"name_p4":None,"addr":N1.na.values}
    X1={}
    for v in Xc:
        src_ser=N1.nn if v=="name" else (N1.na if v=="addr" else p4(N1.nn.iloc[s1i]).reindex(N1.index,fill_value=""))
        X1[v]=vecs[v].transform(src_ser.iloc[pp.i1.values].values if v!="name_p4" else p4(N1.nn.iloc[pp.i1.values]).values).tocsr()
    minshared={}
    for v in Xc:
        A=X1[v]; B=Xc[v][remap[pp.cj.values]]
        prod=A.multiply(B).tocsr()
        rows=np.repeat(np.arange(prod.shape[0]),np.diff(prod.indptr))
        ser=pd.Series(df[v][prod.indices]).groupby(rows).min()
        arr=np.full(prod.shape[0],np.inf); arr[ser.index.values]=ser.values; minshared[v]=arr
    Ts=[5,20,50,100,300,1000,3000,10000,np.inf]
    rec={}
    for T in Ts:
        r={}
        for v in Xc: r[v]=float((minshared[v]<=T).mean())
        r["name|addr"]=float(((minshared["name"]<=T)|(minshared["addr"]<=T)).mean()); r["name_p4|addr"]=float(((minshared["name_p4"]<=T)|(minshared["addr"]<=T)).mean())
        r["name&addr"]=float(((minshared["name"]<=T)&(minshared["addr"]<=T)).mean()); r["name_p4&addr"]=float(((minshared["name_p4"]<=T)&(minshared["addr"]<=T)).mean())
        r["name|name_p4|addr"]=float(((minshared["name"]<=T)|(minshared["name_p4"]<=T)|(minshared["addr"]<=T)).mean())
        rec[str(T)]=r
    R["recall_by_df_cutoff"]=rec
    # recall by source for union
    R["recall_union_T300_by_src"]={int(s):float((((minshared["name_p4"]<=300)|(minshared["addr"]<=300))[pp.src.values==s]).mean()) for s in (2,3)}
    # ---- candidate counts on S1 sample
    smp=rng.choice(s1i,M,replace=False)
    Xs={v:(vecs[v].transform(N1.nn.iloc[smp].values) if v=="name" else vecs[v].transform(N1.na.iloc[smp].values) if v=="addr" else vecs[v].transform(p4(N1.nn.iloc[smp]).values)).tocsr() for v in Xc}
    XT={v:Xc[v].T.tocsr() for v in Xc}   # V x N
    def cands(v,T):
        e=(df[v]<=T).astype(np.int8); A=Xs[v].multiply(e).tocsr(); A.eliminate_zeros()
        P=(A.astype(np.float32)@XT[v].astype(np.float32)); P.data[:]=1; return P.tocsr()
    cc={}
    for T in [20,100,300,1000,3000]:
        Pn=cands("name",T); Pp=cands("name_p4",T); Pa=cands("addr",T)
        sets={"name":Pn,"name_p4":Pp,"addr":Pa,"name|addr":(Pn+Pa),"name_p4|addr":(Pp+Pa),"name&addr":Pn.multiply(Pa),"name_p4&addr":Pp.multiply(Pa)}
        cc[str(T)]={}
        for k,P in sets.items():
            P=sp.csr_matrix(P); P.data[:]=1; cnt=np.diff(P.indptr)
            s2c=np.asarray(P[:,np.where(csrc==2)[0]].sum(1)).ravel(); 
            cc[str(T)][k]={"mean":float(cnt.mean()),"median":float(np.median(cnt)),"p90":float(np.quantile(cnt,.9)),"p99":float(np.quantile(cnt,.99)),"max":int(cnt.max()),"zero_cand_pct":float((cnt==0).mean()*100),"mean_S2":float(s2c.mean())}
        print("  T",T,{k:round(v["mean"]) for k,v in cc[str(T)].items()},round(time.time()-t0),flush=True)
    R["candidates_per_S1"]=cc
    res[country]=R
    json.dump(res,open("out/blocking_token.json","w"),indent=1)
print("done",round(time.time()-t0))
