import sys, time, warnings; sys.path.insert(0,'.'); warnings.filterwarnings('ignore')
from research import emcr_lab as L
import pandas as pd, numpy as np
pd.set_option('display.width',220)
D=L.daily_data()
P=dict(roll_windows=(3,5,10))
for w in (5,3):
    base=L.Design(F=126,T=21,signal='rolling',window=w,entry=1.5,exit=0.0,params=P)
    t=time.time(); ic,_=L.pair_ic(D,base); print(f'\n=== IC roll{w} [{time.time()-t:.0f}s]'); print(ic.round(3).head(14).to_string())
    crits=['Distance','Correlation','Cointegration','HalfLife','EMCR',f'FormPnL{w}d',f'EMCRwin{w}d',f'EMCRnet{w}d',f'EMCRsh{w}d',
           f'FormPnL{w}d+Correlation',f'FormPnL{w}d+Cointegration',f'EMCRsh{w}d+Correlation',f'Correlation|FormPnL{w}d']
    rows=[]
    for n,mu in ((20,1),(50,2),(100,3)):
        for c in crits:
            d=L.Design(label=f'r{w}_n{n}',F=126,T=21,signal='rolling',window=w,entry=1.5,exit=0.0,n_pairs=n,max_uses=mu,criterion=c,params=P)
            o,_=L.evaluate(D,d); rows.append(dict(n=n,crit=c,sharpe=o['sharpe'],ret=o['ann_ret'],gross=o['gross_ret'],vol=o['ann_vol'],turn=o['turnover']))
    print(pd.DataFrame(rows).pivot(index='crit',columns='n',values='sharpe').round(2).sort_values(100,ascending=False).to_string())
    print(pd.DataFrame(rows).pivot(index='crit',columns='n',values='gross').round(3).to_string())
