"""Multi-market scan on DEV folds (2016-2022): same designs and criteria in every market."""
import sys, time, warnings, itertools; sys.path.insert(0,'.'); warnings.filterwarnings('ignore')
from research import emcr_lab as L, markets as M
import pandas as pd, numpy as np
OUT='research/market_scan_dev.csv'
crits=['Distance','Correlation','Cointegration','HalfLife','EMCR','EMCR1','EMCR2','EMCRc','EMCRstable','FormPnL',
       'EMCRnet5d','EMCRwin5d','EMCRnet20d','Correlation|EMCR1','Correlation|EMCR2','EMCR1+Correlation','EMCR2+Cointegration','Distance|EMCR']
designs=[dict(F=252,T=63,signal='static',window=20,entry=2.0,exit=0.0),
         dict(F=252,T=126,signal='static',window=20,entry=2.5,exit=0.5),
         dict(F=126,T=63,signal='static',window=20,entry=2.0,exit=0.5),
         dict(F=126,T=21,signal='rolling',window=5,entry=1.5,exit=0.0),
         dict(F=126,T=21,signal='rolling',window=20,entry=2.0,exit=0.0)]
rows=[]; t0=time.time()
for mk in sys.argv[1:] or list(M.MARKETS):
    D,cfg=M.market(mk)
    for dz in designs:
        for c in crits:
            for n,mu in ((20,1),(50,2)):
                des=L.Design(label=f'MS_{mk}_F{dz["F"]}T{dz["T"]}_{dz["signal"]}{dz["window"]}_e{dz["entry"]}x{dz["exit"]}_n{n}',
                             top_k=cfg['top_k'],n_pairs=n,max_uses=mu,criterion=c,borrow_pa=cfg['borrow'],params=cfg['params'],**dz)
                o,net=L.evaluate(D,des)
                rows.append(dict(market=mk,**dz,n=n,crit=c,sharpe=o['sharpe'],ret=o['ann_ret'],vol=o['ann_vol'],gross=o['gross_ret'],turn=o['turnover']))
        print(f'{mk} {dz} done [{time.time()-t0:.0f}s]', flush=True)
        pd.DataFrame(rows).to_csv(OUT,index=False)
print('ALL DONE', flush=True)
