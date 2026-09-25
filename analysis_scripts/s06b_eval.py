import pandas as pd, numpy as np, json
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
F=pd.read_parquet("out/sim_features.parquet")
feats=[c for c in F.columns if c.startswith(("n_","a_"))]
print(F.kind.value_counts())
P=F[F.kind=="positive"]
res={"counts":F.kind.value_counts().to_dict()}
# distribution table
tab=F.groupby("kind")[feats].mean().T.round(3); print(tab)
med=F.groupby("kind")[feats].median().T.round(3)
res["mean_by_kind"]=tab.to_dict(); res["median_by_kind"]=med.to_dict()
q=P[feats].quantile([.05,.25,.5,.75]).T.round(3); res["positive_quantiles"]=q.to_dict(); print(q)
# AUC per feature vs each negative type (NaN->-1 i.e. missing address treated as lowest)
def auc(sub,f):
    y=(sub.kind=="positive").values; x=sub[f].fillna(-1).values; return roc_auc_score(y,x)
aucs={}
for neg in [k for k in F.kind.unique() if k!="positive"]+["ALL_neg"]:
    sub=F[(F.kind=="positive")|((F.kind!="positive") if neg=="ALL_neg" else (F.kind==neg))]
    aucs[neg]={f:round(auc(sub,f),4) for f in feats}
A=pd.DataFrame(aucs); print(A); res["auc"]=A.to_dict()
# latin-only positives (exclude native script names) 
lat=F[~F.name_native]
aucs2={}
for neg in [k for k in F.kind.unique() if k!="positive"]:
    sub=lat[(lat.kind=="positive")|(lat.kind==neg)]; aucs2[neg]={f:round(auc(sub,f),4) for f in feats}
A2=pd.DataFrame(aucs2); res["auc_latin_names_only"]=A2.to_dict(); print("LATIN ONLY\n",A2)
# best-threshold F0.5 for a few single features vs ALL negatives (balanced sample => report precision at fixed recall)
def best_f05(sub,f):
    y=(sub.kind=="positive").values; x=sub[f].fillna(-1).values
    ths=np.unique(np.quantile(x,np.linspace(0,1,201))); best=(0,None,None,None)
    for t in ths:
        p=x>=t; tp=(p&y).sum(); fp=(p&~y).sum(); fn=(~p&y).sum()
        if tp==0: continue
        pr=tp/(tp+fp); rc=tp/(tp+fn); f=1.25*pr*rc/(.25*pr+rc)
        if f>best[0]: best=(f,t,pr,rc)
    return best
bf={}
for f in ["n_tsort","n_tset","n_jacc","n_lev","n_jw","n_tfidf_word","n_tfidf_char","a_tsort","a_jacc","a_tfidf_word","a_tfidf_char","a_c3jacc"]:
    bf[f]={neg:[round(float(v),3) if v is not None else None for v in best_f05(F[(F.kind=="positive")|(F.kind==neg)],f)] for neg in ["random_same_country","hard_same_block(country+name[:4])","hard_same_name_diff_entity","hard_same_addr_diff_entity"]}
res["best_f05_single_feature(F,thr,prec,rec)"]=bf; print(json.dumps(bf,indent=0))
# combined diagnostic model (NOT final): HGB on all features, negatives = mix, evaluate on hold-out, by neg type
X=F[feats+["addr_empty","name_native"]].astype(float); y=(F.kind=="positive").astype(int)
Xtr,Xte,ytr,yte,ktr,kte=train_test_split(X,y,F.kind,test_size=0.3,random_state=0,stratify=F.kind)
w=np.where(ytr==1,1.0,(ytr==1).sum()/(ytr==0).sum())
m=HistGradientBoostingClassifier(max_iter=200,learning_rate=0.1).fit(Xtr,ytr,sample_weight=w)
pr=m.predict_proba(Xte)[:,1]
diag={}
for neg in [k for k in F.kind.unique() if k!="positive"]:
    mk=(kte=="positive")|(kte==neg); diag[neg]={"auc":round(roc_auc_score(yte[mk],pr[mk]),4)}
    for thr in (0.5,0.8,0.9,0.95):
        p=pr[mk]>=thr; yy=yte[mk].values==1; tp=(p&yy).sum(); fp=(p&~yy).sum(); fn=(~p&yy).sum()
        diag[neg][f"thr{thr}"]={"precision":round(tp/max(1,tp+fp),4),"recall":round(tp/max(1,tp+fn),4)}
mk=np.ones(len(yte),bool); diag["ALL"]={"auc":round(roc_auc_score(yte,pr),4)}
# recall breakdown among positives at 0.9 by segment
Pt=Xte[yte==1].copy(); Pt["pred"]=pr[yte==1]>=0.9; Pt["src"]=F.loc[Pt.index,"src"]; Pt["country"]=F.loc[Pt.index,"country"]
diag["positive_recall@0.9_by"]={"src":Pt.groupby("src").pred.mean().round(3).to_dict(),"country":Pt.groupby("country").pred.mean().round(3).to_dict(),
   "native_script_name":Pt.groupby("name_native").pred.mean().round(3).to_dict(),"addr_empty":Pt.groupby("addr_empty").pred.mean().round(3).to_dict()}
res["diagnostic_hgb"]=diag; print(json.dumps(diag,indent=1))
# name-alone vs address-alone vs both: fraction of positives with name sim high / addr sim high
P=F[F.kind=="positive"]
res["positives_signal_coverage"]={"name_tsort>=0.8":round(float((P.n_tsort>=.8).mean()),4),"addr_tsort>=0.8":round(float((P.a_tsort>=.8).mean()),4),
  "name>=0.8_or_addr>=0.8":round(float(((P.n_tsort>=.8)|(P.a_tsort>=.8)).mean()),4),"neither>=0.8":round(float(((P.n_tsort<.8)&~(P.a_tsort>=.8)).mean()),4),
  "name<0.5_and_addr>=0.7":round(float(((P.n_tsort<.5)&(P.a_tsort>=.7)).mean()),4),"name>=0.8_addr<0.5_or_empty":round(float(((P.n_tsort>=.8)&~(P.a_tsort>=.5)).mean()),4),
  "name<0.5_and_addr<0.5_or_empty(hopeless)":round(float(((P.n_tsort<.5)&~(P.a_tsort>=.5)).mean()),4)}
print(res["positives_signal_coverage"])
# same-name-diff-entity: how confusing?
hn=F[F.kind=="hard_same_name_diff_entity"]; res["hard_same_name_stats"]={"name_tsort_mean":round(float(hn.n_tsort.mean()),3),"addr_tsort_mean":round(float(hn.a_tsort.mean()),3),"addr_jacc_median":round(float(hn.a_jacc.median()),3)}
ha=F[F.kind=="hard_same_addr_diff_entity"]; res["hard_same_addr_stats"]={"name_tsort_mean":round(float(ha.n_tsort.mean()),3),"addr_tsort_mean":round(float(ha.a_tsort.mean()),3)}
json.dump(res,open("out/similarity.json","w"),indent=1,default=str)
