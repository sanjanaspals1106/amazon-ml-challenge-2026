import pandas as pd, csv, os, time
D="/Users/sanjana1106/Desktop/MLChallenge/student_resource/dataset"
files={"train_source1":"train","train_source2":"train","train_source3":"train","train_ground_truth":"train",
       "test_source1":"test","test_source2":"test","test_source3":"test"}
for f,sp in files.items():
    t=time.time()
    df=pd.read_csv(f"{D}/{sp}/{f}.tsv",sep="\t",dtype=str,keep_default_na=False,na_filter=False,quoting=csv.QUOTE_NONE,encoding="utf-8",encoding_errors="strict")
    df.to_parquet(f"cache/{f}.parquet",index=False)
    print(f,df.shape,round(time.time()-t,1),"s",round(df.memory_usage(deep=True).sum()/1e6),"MB",flush=True)
