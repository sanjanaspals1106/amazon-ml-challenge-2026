import pandas as pd, numpy as np, re, json
from rapidfuzz import fuzz
exec(open("s07c_keys.py").read().split("def keyfuncs")[0].split("t0=time.time()")[1].split("N1=")[0]) if False else None
STATES="alabama alaska arizona arkansas california colorado connecticut delaware florida georgia hawaii idaho illinois indiana iowa kansas kentucky louisiana maine maryland massachusetts michigan minnesota mississippi missouri montana nebraska nevada new\\ hampshire new\\ jersey new\\ mexico new\\ york north\\ carolina north\\ dakota ohio oklahoma oregon pennsylvania rhode\\ island south\\ carolina south\\ dakota tennessee texas utah vermont virginia washington west\\ virginia wisconsin wyoming district\\ of\\ columbia".split()
CODES="al ak az ar ca co ct de fl ga hi id il in ia ks ky la me md ma mi mn ms mo mt ne nv nh nj nm ny nc nd oh ok or pa ri sc sd tn tx ut vt va wa wv wi wy dc".split()
st_re=re.compile(r"(?i)(?:^|[\s,])(?:"+"|".join(s.replace("\\ ","\\s") for s in STATES)+"|"+"|".join(c for c in CODES)+r")(?=$|[\s,])")
pos=pd.read_parquet("cache/pos_pairs.parquet"); s1=pd.read_parquet("cache/train_source1.parquet")
res={}
for src in (2,3):
    d=pd.read_parquet(f"cache/train_source{src}.parquet")
    p=pos[pos.src==src].sample(400000,random_state=1)
    a1=s1.business_address.values[p.i1.values]; a2=d.business_address.values[p.j.values]; c=s1.country.values[p.i1.values]
    m=(c=="US")&(a2!=""); a1,a2=a1[m],a2[m]
    A1=pd.Series(a1); A2=pd.Series(a2)
    hs1=A1.str.contains(st_re); hs2=A2.str.contains(st_re)
    def city(a):
        comps=[x.strip() for x in a.split(",")]
        comps=[x for x in comps if not re.search(r"\d",x) and not re.fullmatch(r"(?i)"+"|".join(CODES+[s.replace('\\ ',' ') for s in STATES]),x)]
        return comps[-1] if comps else ""
    c1=A1.map(city)
    inn=np.array([ (cc!="" and (cc.lower() in b.lower() or fuzz.partial_ratio(cc.lower(),b.lower())>=85)) for cc,b in zip(c1.values,a2)])
    exact=np.array([ (cc!="" and cc.lower() in b.lower()) for cc,b in zip(c1.values,a2)])
    res[f"US_S{src}"]={"n":int(len(A1)),"S1_has_state_pct":round(100*float(hs1.mean()),2),"match_missing_state_when_S1_has_pct":round(100*float((hs1&~hs2).mean()/hs1.mean()),2),
        "S1_city_found_exact_in_match_pct":round(100*float(exact.mean()),2),"S1_city_missing_or_misspelled_in_match_(fuzzy85)_pct":round(100*float((~inn).mean()),2),
        "city_present_but_misspelled_pct":round(100*float((inn&~exact).mean()),2),
        "match_has_zip_after_state_pct":round(100*float(A2.str.contains(r"(?i),\s*[A-Z]{2}\s+\d{5}(?:-\d{4})?$").mean()),3),"S1_has_zip_after_state_pct":round(100*float(A1.str.contains(r"(?i),\s*[A-Z]{2}\s+\d{5}(?:-\d{4})?$").mean()),3),
        "match_has_street_number_pct":round(100*float(A2.str.contains(r"\d").mean()),2),"S1_has_street_number_pct":round(100*float(A1.str.contains(r"\d").mean()),2)}
    print(res[f"US_S{src}"])
# India: PIN
for src in (1,):
    pass
for name in ("train_source1","train_source2","train_source3","test_source1","test_source2","test_source3"):
    d=pd.read_parquet(f"cache/{name}.parquet",columns=["business_address","country"]).sample(300000,random_state=1)
    x={}
    for c in d.country.unique():
        s=d[d.country==c].business_address
        x[c]={"PIN_like_6digit_token_pct":round(100*float(s.str.contains(r"(?<![\d/-])[1-9]\d{2}\s?\d{3}(?![\d/-])").mean()),3),
              "5digit_after_comma_or_space_at_end_pct":round(100*float(s.str.contains(r"(?:,|\s)\d{5}\s*$").mean()),3),
              "fr_postcode_before_city_pct":round(100*float(s.str.contains(r"\b\d{5}\s+[A-ZÉ][a-zé]+").mean()),3)}
    res.setdefault("postal_code_presence",{})[name]=x
print(json.dumps(res["postal_code_presence"],indent=0))
json.dump(res,open("out/addr_parts.json","w"),indent=1)
