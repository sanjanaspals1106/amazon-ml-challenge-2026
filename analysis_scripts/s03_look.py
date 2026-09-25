import pandas as pd, numpy as np
C="cache"
s1=pd.read_parquet(f"{C}/train_source1.parquet"); s2=pd.read_parquet(f"{C}/train_source2.parquet"); s3=pd.read_parquet(f"{C}/train_source3.parquet")
pos=pd.read_parquet(f"{C}/pos_pairs.parquet")
rng=np.random.default_rng(7)
grp=pos.groupby("i1")
ids=rng.choice(pos.i1.unique(),40,replace=False)
for i in ids:
    r=s1.iloc[i]; print("\nS1",r.business_name,"|",r.business_address,"|",r.country)
    for _,p in pos[pos.i1==i].iterrows():
        x=(s2 if p.src==2 else s3).iloc[p.j]; print("   S%d"%p.src,x.business_name,"|",x.business_address,"|",x.country)
