import pandas as pd, numpy as np, json, re, time
t0=time.time()
N1=pd.read_parquet("cache/norm_train_source1.parquet"); N2=pd.read_parquet("cache/norm_train_source2.parquet"); N3=pd.read_parquet("cache/norm_train_source3.parquet")
pos=pd.read_parquet("cache/pos_pairs.parquet")
STATES={"al":"alabama","ak":"alaska","az":"arizona","ar":"arkansas","ca":"california","co":"colorado","ct":"connecticut","de":"delaware","fl":"florida","ga":"georgia","hi":"hawaii","id":"idaho","il":"illinois","in":"indiana","ia":"iowa","ks":"kansas","ky":"kentucky","la":"louisiana","me":"maine","md":"maryland","ma":"massachusetts","mi":"michigan","mn":"minnesota","ms":"mississippi","mo":"missouri","mt":"montana","ne":"nebraska","nv":"nevada","nh":"new hampshire","nj":"new jersey","nm":"new mexico","ny":"new york","nc":"north carolina","nd":"north dakota","oh":"ohio","ok":"oklahoma","or":"oregon","pa":"pennsylvania","ri":"rhode island","sc":"south carolina","sd":"south dakota","tn":"tennessee","tx":"texas","ut":"utah","vt":"vermont","va":"virginia","wa":"washington","wv":"west virginia","wi":"wisconsin","wy":"wyoming","dc":"district of columbia"}
NAME2CODE={v:k for k,v in STATES.items()}
st_re=re.compile(r"\b("+"|".join(sorted(list(NAME2CODE)+list(STATES),key=len,reverse=True))+r")\b")
def state(a):
    m=st_re.findall(a)
    if not m: return ""
    x=m[-1]; return NAME2CODE.get(x,x)
def keyfuncs(df,ctry):
    nn=df.nn; na=df.na
    tok=nn.str.split()
    k={}
    k["name_first_token"]=tok.str[0].fillna("")
    k["name_first3chars(nospace)"]=nn.str.replace(" ","").str[:3]
    k["name_first4chars(nospace)"]=nn.str.replace(" ","").str[:4]
    k["name_first6chars(nospace)"]=nn.str.replace(" ","").str[:6]
    k["name_first_token[:4]"]=tok.str[0].fillna("").str[:4]
    k["name_min_token_alpha"]=tok.map(lambda t:min(t) if t else "")
    k["name_longest_token"]=tok.map(lambda t:max(t,key=len) if t else "")
    k["name_sorted_first2tokens"]=tok.map(lambda t:" ".join(sorted(t)[:2]))
    hn=na.str.extract(r"(\d+)")[0].str.lstrip("0").fillna("")
    k["addr_first_number"]=hn
    k["addr_first_number+name_first3"]=hn+"|"+nn.str.replace(" ","").str[:3]
    k["addr_first_number+name_first_token[:4]"]=hn+"|"+tok.str[0].fillna("").str[:4]
    return k
res={}
for ctry in ("US","India"):
    a=N1[N1.country==ctry]; i1=a.index.values
    b=pd.concat([N2[N2.country==ctry],N3[N3.country==ctry]])
    ka=keyfuncs(a,ctry); kb=keyfuncs(b,ctry)
    if ctry=="US":
        ka["state"]=a.na.map(state); kb["state"]=b.na.map(state)
        ka["state+name_first3"]=ka["state"]+"|"+ka["name_first3chars(nospace)"]; kb["state+name_first3"]=kb["state"]+"|"+kb["name_first3chars(nospace)"]
        ka["state+addr_first_number"]=ka["state"]+"|"+ka["addr_first_number"]; kb["state+addr_first_number"]=kb["state"]+"|"+kb["addr_first_number"]
    # positives (per-country)
    p=pos[pos.i1.isin(i1)].sample(200000,random_state=1)
    n2=len(N2)
    R={"N1":len(a),"N23":len(b)}
    # full-frame key arrays for pos lookups
    for name in ka:
        A=pd.Series(ka[name].values,index=a.index); 
        Ball=pd.Series(kb[name].values,index=np.concatenate([N2.index[N2.country==ctry].values,N3.index[N3.country==ctry].values+n2]))
        vc1=A.value_counts(); vc2=Ball.value_counts()
        valid=(vc1.index!="")&(~vc1.index.str.startswith("|"))&(~vc1.index.str.endswith("|"))
        vc1v=vc1[valid]; joined=vc1v.to_frame("n1").join(vc2.rename("n2"),how="left").fillna(0)
        cand_mean=float((joined.n1*joined.n2).sum()/len(a))
        # candidate distribution per S1: value n2 of its key
        per=A.map(vc2).fillna(0)
        per[~A.isin(vc1v.index)]=0
        cj=np.where(p.src.values==2,p.j.values,p.j.values+n2)
        ka_p=A.reindex(p.i1.values).values; kb_p=Ball.reindex(cj).values
        rec=float(((ka_p==kb_p)&(ka_p!="")&pd.Series(ka_p).map(lambda x:not(x.startswith("|") or x.endswith("|"))).values).mean())
        R[name]={"recall":round(rec,4),"cands_mean":round(cand_mean,1),"cands_median":float(per.median()),"cands_p90":float(per.quantile(.9)),"cands_p99":float(per.quantile(.99)),"cands_max":int(per.max()),"s1_with_zero_cands_pct":round(float((per==0).mean()*100),2),"n_distinct_keys_S1":int(len(vc1v))}
        print(ctry,name,R[name],round(time.time()-t0),flush=True)
    res[ctry]=R
json.dump(res,open("out/blocking_keys.json","w"),indent=1)
