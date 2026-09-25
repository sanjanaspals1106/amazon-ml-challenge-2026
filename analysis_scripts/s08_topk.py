import pandas as pd, numpy as np, scipy.sparse as sp, json, time, re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import HistGradientBoostingClassifier
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein, JaroWinkler
t0=time.time(); rng=np.random.default_rng(21)
N1=pd.read_parquet("cache/norm_train_source1.parquet"); N2=pd.read_parquet("cache/norm_train_source2.parquet"); N3=pd.read_parquet("cache/norm_train_source3.parquet")
pos=pd.read_parquet("cache/pos_pairs.parquet"); gc=pd.read_parquet("cache/gt_counts.parquet")
CORP=pd.concat([N2.assign(src=2),N3.assign(src=3)],ignore_index=True); n2=len(N2); pos["cj"]=np.where(pos.src==2,pos.j,pos.j+n2)
truth=pos.groupby("i1").cj.apply(np.array)
INDIC=re.compile(r"[ऀ-෿]"); K=100; DFMAX=20000; KS=[1,3,5,10,20,50,100]
def p4(s): return s.map(lambda x:" ".join(t[:4] for t in x.split() if len(t)>=3))
def jac(a,b):
    x,y=set(a.split()),set(b.split()); return len(x&y)/len(x|y) if (x|y) else 0
def cj3(a,b):
    a,b=a.replace(" ",""),b.replace(" ",""); x={a[i:i+3] for i in range(len(a)-2)}; y={b[i:i+3] for i in range(len(b)-2)}
    return len(x&y)/len(x|y) if (x|y) else 0
def hn(s):
    m=re.search(r"\d+",s); return m.group(0).lstrip("0") if m else None
rows=[]; meta=[]
singleton=(gc.n.values==0); S1_single=set(np.where(singleton)[0])
# gc rows are in gt order not S1 order -> map by id
gcm=gc.set_index("sid").n; n_true=gcm.reindex(pd.read_parquet("cache/train_source1.parquet").entity_id.values).values
for country in ("US","India"):
    cidx=np.where(CORP.country.values==country)[0]; C=CORP.iloc[cidx].reset_index(drop=True); remap=np.full(len(CORP),-1); remap[cidx]=np.arange(len(cidx))
    Cn,Cp,Ca=C.nn.values,p4(C.nn).values,C.na.values
    vec={}; XT={}; w={}
    for v,arr in (("n",Cn),("p",Cp),("a",Ca)):
        cv=CountVectorizer(token_pattern=r"\S+",binary=True,dtype=np.float32,lowercase=False); X=cv.fit_transform(arr).tocsr()
        df=np.asarray(X.sum(0)).ravel(); idf=np.log(len(C)/df).astype(np.float32); idf[df>DFMAX]=0
        vec[v]=cv; XT[v]=X.T.tocsr(); w[v]=idf*(0.5 if v=="p" else 1.0)
    print(country,"built",round(time.time()-t0),flush=True)
    s1i=np.where(N1.country.values==country)[0]
    rnd=rng.choice(s1i,2400,replace=False); sing=rng.choice(np.intersect1d(s1i,np.where(n_true==0)[0]),600,replace=False)
    smp=np.concatenate([rnd,sing]); grp=np.array(["A"]*1200+["B"]*1200+["S"]*600)
    for st in range(0,len(smp),100):
        ids=smp[st:st+100]
        P=None
        for v,src in (("n",N1.nn.values[ids]),("p",p4(pd.Series(N1.nn.values[ids])).values),("a",N1.na.values[ids])):
            A=vec[v].transform(src).tocsr().multiply(w[v]).tocsr(); M=A@XT[v]; P=M if P is None else P+M
        P=P.tocsr()
        for r,i in enumerate(ids):
            a,b=P.indptr[r],P.indptr[r+1]; idx=P.indices[a:b]; val=P.data[a:b]
            if len(idx)>K: s=np.argpartition(-val,K)[:K]; idx,val=idx[s],val[s]
            o=np.argsort(-val); idx,val=idx[o],val[o]
            tr=set(remap[truth[i]]) if i in truth.index else set()
            g=grp[st+r]; mx=val[0] if len(val) else 1
            rows.append((country,int(i),g,idx,val,mx,tr))
    print(country,"scored",round(time.time()-t0),flush=True)
    # features
    globals().setdefault("CDATA",{})[country]=(Cn,Ca,C.src.values)
# --- recall@K
res={"K":K,"DFMAX":DFMAX}
rk={}
for country in ("US","India"):
    for gname in ("random","singletons"):
        sel=[r for r in rows if r[0]==country and ((r[2] in "AB") if gname=="random" else r[2]=="S")]
        hit={k:0 for k in KS}; tot=0; full={k:0 for k in KS}; nz=0; nc=[]
        for _,i,g,idx,val,mx,tr in sel:
            nc.append(len(idx))
            if not tr: continue
            nz+=1; ranks=np.where(np.isin(idx,list(tr)))[0]
            for k in KS:
                h=(ranks<k).sum(); hit[k]+=h; tot+=0
            tot+=len(tr)
            for k in KS: full[k]+=int((ranks<k).sum()==len(tr))
        if gname=="random": rk[country]={"S1_sampled":len(sel),"S1_with_matches":nz,"true_matches":tot,"cands_returned_mean":float(np.mean(nc)),
            "recall_at_K":{k:round(hit[k]/tot,4) for k in KS},"all_matches_within_K_pct_of_entities":{k:round(full[k]/nz,4) for k in KS}}
res["recall_at_K"]=rk
# --- candidate features & diagnostic model
def featrow(country,i,j,rank,score,mx):
    Cn,Ca,Cs=CDATA[country]; n1,a1=N1.nn.values[i],N1.na.values[i]; n2_,a2=Cn[j],Ca[j]
    ae=a2==""
    f=[fuzz.ratio(n1,n2_)/100,fuzz.token_sort_ratio(n1,n2_)/100,fuzz.token_set_ratio(n1,n2_)/100,fuzz.partial_ratio(n1,n2_)/100,JaroWinkler.similarity(n1,n2_),Levenshtein.normalized_similarity(n1,n2_),jac(n1,n2_),cj3(n1,n2_),
       np.nan if ae else fuzz.token_sort_ratio(a1,a2)/100,np.nan if ae else fuzz.token_set_ratio(a1,a2)/100,np.nan if ae else jac(a1,a2),np.nan if ae else cj3(a1,a2),
       np.nan if ae else float(hn(a1) is not None and hn(a1)==hn(a2)),float(ae),float(bool(INDIC.search(n2_))),score,score/mx if mx else 0,rank,Cs[j]==3]
    return f
cols=["n_ratio","n_tsort","n_tset","n_partial","n_jw","n_lev","n_jacc","n_c3","a_tsort","a_tset","a_jacc","a_c3","a_hn","a_empty","n_native","score","rel_score","rank","src3"]
X=[];Y=[];key=[]
for (country,i,g,idx,val,mx,tr) in rows:
    for rk_,(j,s) in enumerate(zip(idx,val)):
        X.append(featrow(country,i,j,rk_,float(s),float(mx))); Y.append(int(j in tr)); key.append((country,i,g,int(j)))
X=np.array(X,dtype=float); Y=np.array(Y); key=pd.DataFrame(key,columns=["country","i","g","j"])
print("features",X.shape,round(time.time()-t0),flush=True)
tr=(key.g=="A").values
model=HistGradientBoostingClassifier(max_iter=250,learning_rate=0.08,random_state=0).fit(X[tr],Y[tr])
p=model.predict_proba(X)[:,1]; key["p"]=p; key["y"]=Y
key.to_parquet("out/topk_scored.parquet")
# per-entity F0.5
def f05(pred,truth):
    if not truth: return 1.0 if not pred else 0.0
    if not pred: return 0.0
    tp=len(pred&truth); 
    if tp==0: return 0.0
    P=tp/len(pred); R=tp/len(truth); return 1.25*P*R/(.25*P+R)
truthsets={(r[0],r[1]):r[6] for r in rows}
# use corpus-local idx sets: key.j are corpus-local indices; truth sets are corpus-local too
ev=key[key.g!="A"]
out={}
for thr in [0.3,0.5,0.6,0.7,0.8,0.9,0.95]:
    sc={"random":[], "singletons":[], "nonsingleton":[]}; per_country={"US":[],"India":[]}
    for (country,i,g),d in ev.groupby(["country","i","g"]):
        pred=set(d.j[d.p>=thr]); t=truthsets[(country,i)]; s=f05(pred,t)
        if g in("B",): 
            sc["random"].append(s); per_country[country].append(s)
            (sc["singletons"] if not t else sc["nonsingleton"]).append(s)
        elif g=="S": sc["singletons"].append(s)
    out[thr]={"macro_F05_random_S1_sample":round(float(np.mean(sc["random"])),4),"US":round(float(np.mean(per_country["US"])),4),"India":round(float(np.mean(per_country["India"])),4),
              "singleton_score(all sampled singletons)":round(float(np.mean(sc["singletons"])),4),"nonsingleton_score":round(float(np.mean(sc["nonsingleton"])),4),"n_random":len(sc["random"]),"n_singletons":len(sc["singletons"])}
    print(thr,out[thr],flush=True)
res["diagnostic_baseline"]=out
# ceiling given candidates (oracle within top-100)
cs=[]
for (country,i,g),d in ev[ev.g=="B"].groupby(["country","i","g"]):
    t=truthsets[(country,i)]; pred=set(d.j[d.y==1]); cs.append(f05(pred,t))
res["oracle_ceiling_top100"]=round(float(np.mean(cs)),4)
# singleton danger: max p among false candidates for singletons vs non-singletons
dang={}
for label,mask in (("singletons",(key.g=="S")|((key.g=="B")&(key.y.groupby([key.country,key.i]).transform("sum")==0))),("non_singleton_random_B_false_cands",(key.g=="B")&(key.y.groupby([key.country,key.i]).transform("sum")>0))):
    d=key[mask&(key.y==0)&(key.g!="A")]; mp=d.groupby(["country","i"]).p.max()
    dang[label]={"n_entities":int(len(mp)),**{f"P(max false p>={t})":round(float((mp>=t).mean()),4) for t in (0.5,0.8,0.9,0.95)}}
res["singleton_danger"]=dang
# distribution of p among true matches vs false cands (eval)
e=ev; res["p_quantiles"]={"true":e.p[e.y==1].quantile([.05,.1,.25,.5]).round(3).to_dict(),"false":e.p[e.y==0].quantile([.9,.99,.999,.9999]).round(4).to_dict()}
json.dump(res,open("out/topk.json","w"),indent=1)
print(json.dumps(res["recall_at_K"],indent=1)); print(res["singleton_danger"],res["oracle_ceiling_top100"],res["p_quantiles"])
