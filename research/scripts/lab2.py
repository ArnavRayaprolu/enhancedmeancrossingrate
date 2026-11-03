import sys, time; sys.path.insert(0,'.')
from research import emcr_lab as L
import pandas as pd, numpy as np
pd.set_option('display.width',200)
D=L.daily_data()
for F,T,w,e in [(126,21,5,1.5),(126,21,10,1.5),(126,21,20,1.5),(252,63,20,2.0)]:
    des=L.Design(F=F,T=T,signal='rolling',window=w,entry=e,exit=0.0)
    t=time.time(); ic,_=L.pair_ic(D,des); print(f'\n=== F{F} T{T} roll{w} e{e}  [{time.time()-t:.0f}s]  mean pair net y:'); print(ic.round(3).head(12).to_string())
