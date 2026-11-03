import sys, time, warnings; sys.path.insert(0,'.'); warnings.filterwarnings('ignore')
from research import emcr_lab as L
import pandas as pd, numpy as np
pd.set_option('display.width',220)
D=L.daily_data()
P=dict(roll_windows=(3,5,10), half_bps=2.0)
for w,e in ((5,1.5),(5,2.0),(10,2.0),(20,2.0)):
    base=L.Design(F=126,T=21,signal='rolling',window=w,entry=e,exit=0.0,params=P)
    t=time.time(); ic,_=L.pair_ic(D,base); print(f'\n=== IC roll{w} e{e} [{time.time()-t:.0f}s]'); print(ic.round(3).head(10).to_string())
    crits=['Distance','Correlation','Cointegration','HalfLife','EMCR','FormPnL3d','EMCRnet3d','EMCRsh3d','FormPnL5d','EMCRnet5d','EMCRsh5d',
           'EMCRnet3d+Correlation','EMCRnet3d+Cointegration','Correlation|EMCRnet3d']
    rows=[]
    for n,mu in ((20,1),(50,2),(100,3)):
        for c in crits:
            d=L.Design(label=f'flat_r{w}e{e}_n{n}',F=126,T=21,signal='rolling',window=w,entry=e,exit=0.0,n_pairs=n,max_uses=mu,criterion=c,params=P)
            o,net=L.evaluate(D,d); rows.append(dict(n=n,crit=c,sharpe=o['sharpe'],gross=o['gross_ret'],ret=o['ann_ret'],turn=o['turnover']))
    df=pd.DataFrame(rows)
    print(df.pivot(index='crit',columns='n',values='sharpe').round(2).sort_values(50,ascending=False).to_string())
    print(df.pivot(index='crit',columns='n',values='ret').round(3).sort_values(50,ascending=False).to_string())
    print('turnover', df.groupby('n').turn.mean().round(0).to_dict())
