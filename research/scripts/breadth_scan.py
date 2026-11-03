"""Breadth / risk-balancing scan, full walk-forward 2016-2026, large caps (hourly-derived daily data)."""
import sys, time, warnings; sys.path.insert(0,'.'); warnings.filterwarnings('ignore')
from research import emcr_lab as L
import pandas as pd
same=None if sys.argv[1]=='all' else sys.argv[1]
OUT=f'research/breadth_scan_{sys.argv[1]}.csv'
D=L.daily_data()
P=dict(half_bps=2.0,roll_windows=(5,),same=same)
crits=['Correlation','Cointegration','Distance','EMCRk','Correlation|EMCRkcyc','Correlation|EMCRvwcyc','EMCRvwcyc+Correlation','Correlation|EMCR1']
rows=[]; t0=time.time()
for dz in (dict(F=126,T=63,signal='static',entry=2.5,exit=0.5), dict(F=252,T=63,signal='static',entry=2.0,exit=0.0)):
    for c in crits:
        for n,mu in ((20,1),(50,2),(100,4),(200,8),(400,16)):
            for wt in ('equal','invvol'):
                des=L.Design(label=f'BR_{sys.argv[1]}_F{dz["F"]}T{dz["T"]}_n{n}_{wt}',top_k=250,n_pairs=n,max_uses=mu,weighting=wt,criterion=c,params=P,**dz)
                o,net=L.evaluate(D,des,part='all')
                rows.append(dict(same=sys.argv[1],**dz,n=n,w=wt,crit=c,sharpe=o['sharpe'],ret=o['ann_ret'],vol=o['ann_vol'],turn=o['turnover']))
    print(f'{sys.argv[1]} {dz} done [{time.time()-t0:.0f}s]', flush=True)
    pd.DataFrame(rows).to_csv(OUT,index=False)
print('ALL DONE', flush=True)
