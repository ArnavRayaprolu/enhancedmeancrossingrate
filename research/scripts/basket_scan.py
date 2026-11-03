"""EMCR basket stat-arb scan, full walk-forward 2016-2026, large caps (hourly-derived daily data)."""
import sys, time, warnings; sys.path.insert(0,'.'); warnings.filterwarnings('ignore')
from research import emcr_lab as L, basket_lab as B
import pandas as pd
D=L.daily_data(); t0=time.time()
rows=[]
for F,T in ((252,63),(126,63),(126,21)):
    prep=B.prepare(D,F,T)
    print(f'prepared F{F} T{T} folds={len(prep)} names/fold={sum(len(p["names"]) for p in prep)/len(prep):.0f} [{time.time()-t0:.0f}s]',flush=True)
    for crit in ['EMCR','EMCR1','EMCR2','EMCRk','R2','ADF','SSD','HalfLife','R2|EMCR1','R2|EMCRk','EMCR1+R2','EMCRk+ADF']:
        for n in (50,100,200):
            for e,x in ((2.0,0.0),(1.5,0.0),(2.0,0.5)):
                o,_=B.evaluate(prep,crit,n,e,x)
                rows.append(dict(F=F,T=T,**o))
    pd.DataFrame(rows).to_csv('research/basket_scan.csv',index=False)
d=pd.DataFrame(rows)
print(d.sort_values('sharpe',ascending=False).head(15).round(3).to_string(index=False))
print(d.groupby('crit').sharpe.agg(['mean','max']).sort_values('max',ascending=False).round(2).to_string())
