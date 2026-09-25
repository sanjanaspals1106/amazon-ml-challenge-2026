import pandas as pd, numpy as np, json, re
from collections import Counter
C="cache"
s1=pd.read_parquet(f"{C}/train_source1.parquet"); s2=pd.read_parquet(f"{C}/train_source2.parquet"); s3=pd.read_parquet(f"{C}/train_source3.parquet")
gt=pd.read_parquet(f"{C}/train_ground_truth.parquet")
R={}
R["gt_rows"]=len(gt); R["gt_unique_s1"]=int(gt.source1_entity_id.nunique())
R["gt_s1_in_source1"]=int(gt.source1_entity_id.isin(s1.entity_id).sum())
R["source1_not_in_gt"]=int((~s1.entity_id.isin(gt.source1_entity_id)).sum())
R["gt_s1_id_pattern_bad"]=int((~gt.source1_entity_id.str.fullmatch(r"S1-\d+")).sum())
m=gt.matched_entity_ids
R["gt_empty"]=int((m=="").sum())
# malformed detection
toks_bad=m[(m!="")&(~m.str.fullmatch(r"(?:S[23]-\d+)(?:,S[23]-\d+)*"))]
R["gt_malformed_match_field"]=int(len(toks_bad)); R["gt_malformed_examples"]=toks_bad.head(5).tolist()
R["gt_has_space_after_comma"]=int(m.str.contains(", ").sum())
# explode
g=gt[m!=""].copy(); g["ids"]=g.matched_entity_ids.str.split(",")
g["n"]=g.ids.str.len()
ex=g[["source1_entity_id","ids"]].explode("ids").rename(columns={"ids":"mid"}).reset_index(drop=True)
ex["src"]=ex.mid.str[:2]
R["total_pairs"]=len(ex); R["pairs_s2"]=int((ex.src=="S2").sum()); R["pairs_s3"]=int((ex.src=="S3").sum())
R["dup_matched_ids_globally"]=int(ex.mid.duplicated().sum()); R["unique_matched_ids"]=int(ex.mid.nunique())
R["dup_within_row"]=int((g.ids.map(lambda x: len(x)!=len(set(x)))).sum())
R["matched_s2_in_source2"]=int(ex[ex.src=="S2"].mid.isin(s2.entity_id).sum()); R["matched_s3_in_source3"]=int(ex[ex.src=="S3"].mid.isin(s3.entity_id).sum())
R["matched_s2_unique"]=int(ex[ex.src=="S2"].mid.nunique()); R["matched_s3_unique"]=int(ex[ex.src=="S3"].mid.nunique())
# dup matched ids across multiple S1 entities
d=ex[ex.mid.duplicated(keep=False)].sort_values("mid"); R["mid_shared_across_s1_rows"]=int(len(d)); R["mid_shared_examples"]=d.head(6).values.tolist()
# per-S1 counts
cnt=pd.Series(0,index=gt.source1_entity_id.values)
c_all=ex.groupby("source1_entity_id").size()
c2=ex[ex.src=="S2"].groupby("source1_entity_id").size(); c3=ex[ex.src=="S3"].groupby("source1_entity_id").size()
tot=pd.DataFrame({"n":c_all,"n2":c2,"n3":c3}).reindex(gt.source1_entity_id.values).fillna(0).astype(int)
R["n_s1"]=len(tot); R["zero"]=int((tot.n==0).sum()); R["exactly1"]=int((tot.n==1).sum()); R["multi"]=int((tot.n>1).sum())
R["mean_matches"]=float(tot.n.mean()); R["median"]=float(tot.n.median()); R["max"]=int(tot.n.max()); R["mean_matches_nonzero"]=float(tot.n[tot.n>0].mean())
R["dist_n"]={int(k):int(v) for k,v in tot.n.value_counts().sort_index().items()}
R["dist_n2"]={int(k):int(v) for k,v in tot.n2.value_counts().sort_index().items()}
R["dist_n3"]={int(k):int(v) for k,v in tot.n3.value_counts().sort_index().items()}
R["both_s2_s3"]=int(((tot.n2>0)&(tot.n3>0)).sum()); R["only_s2"]=int(((tot.n2>0)&(tot.n3==0)).sum()); R["only_s3"]=int(((tot.n2==0)&(tot.n3>0)).sum())
R["mean_n2"]=float(tot.n2.mean()); R["mean_n3"]=float(tot.n3.mean()); R["max_n2"]=int(tot.n2.max()); R["max_n3"]=int(tot.n3.max())
# coverage of S2/S3 records
R["s2_rows"]=len(s2); R["s2_matched_frac"]=R["matched_s2_unique"]/len(s2); R["s3_matched_frac"]=R["matched_s3_unique"]/len(s3)
R["s2_unmatched"]=len(s2)-R["matched_s2_unique"]; R["s3_unmatched"]=len(s3)-R["matched_s3_unique"]
# id order in ground truth
R["gt_order_same_as_source1"]=bool((gt.source1_entity_id.values==s1.entity_id.values).all())
R["gt_sorted"]=bool(gt.source1_entity_id.is_monotonic_increasing)
# is the order of ids in list S2 first then S3?
def s2first(x):
    seen3=False
    for t in x:
        if t[1]=="3": seen3=True
        elif seen3: return False
    return True
R["gt_lists_s2_before_s3_all"]=bool(g.ids.map(s2first).all())
print(json.dumps({k:v for k,v in R.items()},indent=1,default=str))
json.dump(R,open("gt_stats.json","w"),indent=1,default=str)
# integer position pairs
p1=pd.Series(np.arange(len(s1)),index=s1.entity_id); p2=pd.Series(np.arange(len(s2)),index=s2.entity_id); p3=pd.Series(np.arange(len(s3)),index=s3.entity_id)
a=p1.reindex(ex.source1_entity_id).values
b=np.where(ex.src=="S2",p2.reindex(ex.mid).values,p3.reindex(ex.mid).values)
print("nan in a",np.isnan(a).sum(),"nan in b",np.isnan(b).sum())
pos=pd.DataFrame({"i1":a.astype("int64"),"j":b.astype("int64"),"src":(ex.src=="S3").astype("int8").values+2})
pos.to_parquet(f"{C}/pos_pairs.parquet",index=False)
tot.assign(sid=tot.index).reset_index(drop=True).to_parquet(f"{C}/gt_counts.parquet")
print(pos.head(), len(pos))
