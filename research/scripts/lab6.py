"""Systematic search on DEV folds: slow (static) designs, same-industry candidates."""
import sys, time, warnings, itertools; sys.path.insert(0,'.'); warnings.filterwarnings('ignore')
from research import emcr_lab as L
import pandas as pd, numpy as np
D=L.daily_data()
crits=['Correlation','Distance','Cointegration','HalfLife','EMCR','EMCR1','EMCR2','EMCRc','EMCRstable','FormPnL',
       'EMCR2+Correlation','EMCR1+Correlation','FormPnL+Correlation','EMCR+Correlation',
       'Correlation|EMCR2','Correlation|EMCR1','Correlation|FormPnL','Correlation|EMCR','Distance|EMCR']
rows=[]; t0=time.time()
for same,(F,T),e,x in itertools.product(('subind','sector'),((252,63),(252,126),(126,63)),(1.5,2.0,2.5),(0.0,0.5)):
    P=dict(roll_windows=(5,), half_bps=2.0, same=same)
    for c in crits:
        for n,mu in ((20,1),(50,2),(100,3)):
            for stop in (0.0,4.0):
                d=L.Design(label=f'S6_{same}_F{F}T{T}_e{e}x{x}_n{n}_s{stop}',F=F,T=T,signal='static',entry=e,exit=x,stop=stop,
                           n_pairs=n,max_uses=mu,criterion=c,params=P)
                o,_=L.evaluate(D,d)
                rows.append(dict(same=same,F=F,T=T,entry=e,exit=x,n=n,stop=stop,crit=c,sharpe=o['sharpe'],ret=o['ann_ret'],vol=o['ann_vol'],gross=o['gross_ret'],turn=o['turnover']))
    print(f'done {same} F{F}T{T} e{e} x{x}  [{time.time()-t0:.0f}s]', flush=True)
    pd.DataFrame(rows).to_csv('/tmp/claude-0/-home-user-enhancedmeancrossingrate/f2391646-d50a-5b7a-a2af-d29f03459ef1/scratchpad/lab6.csv',index=False)
print('ALL DONE', flush=True)
