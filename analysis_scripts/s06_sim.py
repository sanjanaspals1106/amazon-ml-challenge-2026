import pandas as pd, numpy as np, json, re, time
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein, JaroWinkler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import HistGradientBoostingClassifier
from util import *
t0=time.time(); rng=np.random.default_rng(11)
s1=load("train_source1"); pos=load("pos_pairs"); S={2:load("train_source2"),3:load("train_source3")}
# owner arrays (which S1 owns each S2/S3 record; -1 unmatched)
own={k:np.full(len(S[k]),-1,dtype=np.int64) for k in (2,3)}
for k in (2,3):
    p=pos[pos.src==k]; own[k][p.j.values]=p.i1.values
NP=150000
INDIC=re.compile(r"[ऀ-෿]")
# ---- positives
P=pos.iloc[rng.choice(len(pos),NP,replace=False)][["i1","j","src"]].copy(); P["kind"]="positive"
# ---- random same-country negatives (any S2/S3 record incl. unmatched)
def rand_neg(n):
    i1=rng.integers(0,len(s1),n); k=rng.choice([2,3],n); j=np.zeros(n,dtype=np.int64)
    c1=s1.country.values
    for kk in (2,3):
        m=k==kk; d=S[kk]; 
        # sample same-country by rejection: pools per country
        for c in ("US","India"):
            pool=np.where(d.country.values==c)[0]
            mm=m&(c1[i1]==c); j[mm]=rng.choice(pool,mm.sum())
    return pd.DataFrame({"i1":i1,"j":j,"src":k,"kind":"random_same_country"})
N1=rand_neg(NP)
# ---- hard negatives: same block (country + first 4 chars of first name token)
s1n=norm_basic(s1.business_name)
def first4(x): return x.str.split(n=1).str[0].fillna("").str[:4]
s1key=s1.country+"|"+first4(s1n)
H=[]
for kk in (2,3):
    d=S[kk]; pool=np.sort(rng.choice(len(d),1500000,replace=False))
    pk=(d.country.iloc[pool].values+"|"+first4(norm_basic(d.business_name.iloc[pool])).values)
    order=np.argsort(pk,kind="stable"); pk=pk[order]; pj=pool[order]
    smp=rng.choice(len(s1),NP//2,replace=False)
    lo=np.searchsorted(pk,s1key.values[smp],"left"); hi=np.searchsorted(pk,s1key.values[smp],"right")
    ok=hi>lo; smp,lo,hi=smp[ok],lo[ok],hi[ok]
    j=pj[lo+(rng.random(len(lo))*(hi-lo)).astype(int)]
    keep=own[kk][j]!=smp
    H.append(pd.DataFrame({"i1":smp[keep],"j":j[keep],"src":kk,"kind":"hard_same_block(country+name[:4])"}))
H1=pd.concat(H,ignore_index=True)
# ---- same-name-different-entity: S1 a, other S1 b with same normalized name, take a match of b
def twin_neg(keyser,label,n):
    g=pd.DataFrame({"k":keyser.values,"i":np.arange(len(s1))})
    vc=g.k.value_counts(); dup=g[g.k.isin(vc[vc>1].index)&(g.k!="")]
    dup=dup.sort_values("k").reset_index(drop=True)
    st=dup.groupby("k").cumcount().values; sz=dup.groupby("k").k.transform("size").values
    base=np.arange(len(dup))-st
    pick=rng.choice(len(dup),n*3)
    a=dup.i.values[pick]; b=dup.i.values[base[pick]+(rng.random(len(pick))*sz[pick]).astype(int)]
    m=a!=b; a,b=a[m],b[m]
    ps=pos.sort_values("i1").reset_index(drop=True); lo=np.searchsorted(ps.i1.values,b,"left"); hi=np.searchsorted(ps.i1.values,b,"right")
    ok=hi>lo; a,lo,hi=a[ok],lo[ok],hi[ok]
    idx=lo+(rng.random(len(lo))*(hi-lo)).astype(int)
    out=pd.DataFrame({"i1":a,"j":ps.j.values[idx],"src":ps.src.values[idx],"kind":label}).iloc[:n]
    return out
H2=twin_neg(s1n,"hard_same_name_diff_entity",NP//2)
s1a=norm_ext(s1.business_address,ABBR,True); s1a=s1a.where(s1a!="","")
H3=twin_neg(s1a,"hard_same_addr_diff_entity",NP//2)
print("sampled",round(time.time()-t0),flush=True)
ALL=pd.concat([P,N1,H1,H2,H3],ignore_index=True)
# sanity: negatives must not be true pairs
posset=set(zip(pos.i1.values*4+pos.src.values,pos.j.values)) if False else None
own_j=np.where(ALL.src.values==2,0,0)
def is_true(r):
    return own[r[0]][r[1]]==r[2]
bad=np.array([own[s][j]==i for i,j,s in zip(ALL.i1.values,ALL.j.values,ALL.src.values)])
print("true pairs among negatives:",int((bad&(ALL.kind!="positive")).sum()))
ALL=ALL[~(bad&(ALL.kind!="positive"))].reset_index(drop=True)
# ---- features
n1=pd.Series(s1.business_name.values[ALL.i1.values]); a1=pd.Series(s1.address if False else s1.business_address.values[ALL.i1.values])
n2=np.empty(len(ALL),dtype=object); a2=np.empty(len(ALL),dtype=object); c2=np.empty(len(ALL),dtype=object)
for kk in (2,3):
    m=ALL.src.values==kk; n2[m]=S[kk].business_name.values[ALL.j.values[m]]; a2[m]=S[kk].business_address.values[ALL.j.values[m]]
n2=pd.Series(n2); a2=pd.Series(a2)
ne1=norm_ext(n1,ABBR_NAME); ne2=norm_ext(n2,ABBR_NAME); ae1=norm_ext(a1,ABBR,True); ae2=norm_ext(a2,ABBR,True)
F=pd.DataFrame({"kind":ALL.kind.values,"src":ALL.src.values,"country":s1.country.values[ALL.i1.values]})
F["name_native"]=n2.str.contains(INDIC).values; F["addr_empty"]=(a2=="").values
def col(fn,x,y): return np.array([fn(a,b) for a,b in zip(x.values,y.values)])
F["n_ratio"]=col(fuzz.ratio,ne1,ne2)/100; F["n_tsort"]=col(fuzz.token_sort_ratio,ne1,ne2)/100; F["n_tset"]=col(fuzz.token_set_ratio,ne1,ne2)/100
F["n_partial"]=col(fuzz.partial_ratio,ne1,ne2)/100; F["n_lev"]=col(Levenshtein.normalized_similarity,ne1,ne2); F["n_jw"]=col(JaroWinkler.similarity,ne1,ne2)
def jac(a,b):
    x,y=set(a.split()),set(b.split()); return len(x&y)/len(x|y) if (x|y) else 0
def ovl(a,b):
    x,y=set(a.split()),set(b.split()); return len(x&y)/min(len(x),len(y)) if x and y else 0
def cj(a,b,n=3):
    a,b=a.replace(" ",""),b.replace(" ",""); x={a[i:i+n] for i in range(len(a)-n+1)}; y={b[i:i+n] for i in range(len(b)-n+1)}
    return len(x&y)/len(x|y) if (x|y) else 0
F["n_jacc"]=col(jac,ne1,ne2); F["n_overlap"]=col(ovl,ne1,ne2); F["n_c3jacc"]=col(cj,ne1,ne2)
F["a_tsort"]=col(fuzz.token_sort_ratio,ae1,ae2)/100; F["a_tset"]=col(fuzz.token_set_ratio,ae1,ae2)/100; F["a_lev"]=col(Levenshtein.normalized_similarity,ae1,ae2)
F["a_jacc"]=col(jac,ae1,ae2); F["a_overlap"]=col(ovl,ae1,ae2); F["a_c3jacc"]=col(cj,ae1,ae2)
hn=lambda s: s.str.extract(r"(\d+)")[0].str.lstrip("0")
h1,h2=hn(a1),hn(a2); F["a_housenum_eq"]=((h1==h2)&h1.notna()&h2.notna()).values.astype(float)
F.loc[F.addr_empty,["a_tsort","a_tset","a_lev","a_jacc","a_overlap","a_c3jacc","a_housenum_eq"]]=np.nan
print("feats",round(time.time()-t0),flush=True)
# TF-IDF cosine (word, and char 3-gram) fit on random sample of names/addresses across sources
samp_names=pd.concat([norm_ext(s1.business_name.sample(200000,random_state=1),ABBR_NAME)]+[norm_ext(S[k].business_name.sample(200000,random_state=1),ABBR_NAME) for k in (2,3)])
samp_addr=pd.concat([norm_ext(s1.business_address.sample(200000,random_state=1),ABBR,True)]+[norm_ext(S[k].business_address.sample(200000,random_state=1),ABBR,True) for k in (2,3)])
def tfcos(vec,fit,x,y):
    vec.fit(fit); X,Y=vec.transform(x),vec.transform(y); return np.asarray(X.multiply(Y).sum(1)).ravel()
F["n_tfidf_word"]=tfcos(TfidfVectorizer(sublinear_tf=True,min_df=2,token_pattern=r"\S+"),samp_names,ne1,ne2)
F["n_tfidf_char"]=tfcos(TfidfVectorizer(analyzer="char_wb",ngram_range=(3,3),sublinear_tf=True,min_df=3,dtype=np.float32),samp_names,ne1,ne2)
F["a_tfidf_word"]=tfcos(TfidfVectorizer(sublinear_tf=True,min_df=2,token_pattern=r"\S+"),samp_addr,ae1,ae2)
F["a_tfidf_char"]=tfcos(TfidfVectorizer(analyzer="char_wb",ngram_range=(3,3),sublinear_tf=True,min_df=3,dtype=np.float32),samp_addr,ae1,ae2)
F.loc[F.addr_empty,["a_tfidf_word","a_tfidf_char"]]=np.nan
F["exact_name_basic"]=(norm_basic(n1)==norm_basic(n2)).values
F.to_parquet("out/sim_features.parquet")
print("tfidf",round(time.time()-t0),flush=True)
