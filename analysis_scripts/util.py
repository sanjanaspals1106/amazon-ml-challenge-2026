import re, unicodedata, numpy as np, pandas as pd
C="cache"
PUNCT_RE = r"[!-/:-@\[-`{-~‐-―‘-‟…«»।॥·]"
COMB_RE = re.compile(r"[̀-ͯ]")
ABBR = {"pvt":"private","ltd":"limited","corp":"corporation","inc":"incorporated","co":"company","rd":"road","st":"street","ave":"avenue","av":"avenue",
        "dr":"drive","ln":"lane","ct":"court","cir":"circle","blvd":"boulevard","bd":"boulevard","hwy":"highway","pkwy":"parkway","ter":"terrace","pl":"place",
        "apt":"apartment","ste":"suite","fl":"floor","nr":"near","opp":"opposite","bldg":"building","cv":"cove","trl":"trail","sq":"square","mkt":"market",
        "r":"rue","n":"north","s":"south","e":"east","w":"west"}
ABBR_NAME = {k:v for k,v in ABBR.items() if k not in ("r","n","s","e","w","av","bd","dr","st","pl","ct","cv","fl")}  # names: avoid ambiguous ones (Dr/St)
PLACEHOLDER = {"null","na","n","a","nan","none"}
def norm_basic(s):
    """lowercase, ASCII/common-unicode punctuation -> space, collapse whitespace, strip. Keeps accents & native scripts."""
    return s.str.lower().str.replace(PUNCT_RE," ",regex=True).str.replace(r"\s+"," ",regex=True).str.strip()
def fold(s):
    """strip Latin combining diacritics only (NFKD then drop U+0300-036F); Indic vowel signs untouched"""
    return pd.Series([COMB_RE.sub("",unicodedata.normalize("NFKD",x)) if not x.isascii() else x for x in s.values],index=s.index)
def norm_ext(s,abbr=ABBR,addr=False):
    """basic + diacritic fold + '&'->'and' + abbreviation expansion (+ drop NULL/N/A placeholders for addresses)"""
    x=fold(s.str.lower().str.replace("&"," and ",regex=False))
    x=x.str.replace(PUNCT_RE," ",regex=True).str.replace(r"\s+"," ",regex=True).str.strip()
    pat=re.compile(r"\b(" + "|".join(sorted(abbr,key=len,reverse=True)) + r")\b")
    x=x.str.replace(pat,lambda m:abbr[m.group(1)],regex=True)
    if addr:
        x=x.str.replace(r"\b(null|n a|na)\b"," ",regex=True).str.replace(r"\s+"," ",regex=True).str.strip()
    return x
def sort_tokens(s):
    return s.str.split().map(lambda t:" ".join(sorted(t)))
def load(name): return pd.read_parquet(f"{C}/{name}.parquet")
