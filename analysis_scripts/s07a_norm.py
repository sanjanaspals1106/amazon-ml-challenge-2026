import pandas as pd, numpy as np, time, sys
from util import *
t=time.time()
for name in sys.argv[1:]:
    df=load(name); out=[]
    for st in range(0,len(df),1000000):
        c=df.iloc[st:st+1000000]
        out.append(pd.DataFrame({"nn":norm_ext(c.business_name,ABBR_NAME).values,"na":norm_ext(c.business_address,ABBR,True).values,"country":c.country.values}))
    pd.concat(out,ignore_index=True).to_parquet(f"cache/norm_{name}.parquet"); print(name,round(time.time()-t),flush=True)
