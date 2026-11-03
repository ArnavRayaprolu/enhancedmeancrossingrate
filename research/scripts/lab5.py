import sys, time, warnings; sys.path.insert(0,'.'); warnings.filterwarnings('ignore')
from research import emcr_lab as L
import pandas as pd, numpy as np
pd.set_option('display.width',220)
D=L.daily_data()
crits=['Distance','Correlation','Cointegration','HalfLife','EMCR','EMCR1','EMCRstable','FormPnL','EMCRnet5d','EMCRwin5d','FormPnL5d','EMCRnet20d','FormPnL20d',
       'EMCR+Distance','EMCR+Cointegration','EMCRnet5d+Correlation']
for same in ('subind','sector'):
    for (F,T,sig,w,e) in ((126,21,'rolling',5,1.5),(126,21,'rolling',20,2.0),(252,63,'static',20,2.0)):
        P=dict(roll_windows=(5,20), half_bps=2.0, same=same)
        base=L.Design(F=F,T=T,signal=sig,window=w,entry=e,exit=0.0,params=P)
        t=time.time(); ic,_=L.pair_ic(D,base)
        print(f'\n=== {same} F{F} T{T} {sig}{w} e{e} [{time.time()-t:.0f}s] cands/fold={np.mean([len(p["stats"]) for p in L.prepare(D,base)]):.0f}', flush=True)
        print(ic.round(3).head(8).to_string(), flush=True)
        rows=[]
        for n,mu in ((20,1),(50,2)):
            for c in crits:
                d=L.Design(label=f'{same}_F{F}_{sig}{w}_e{e}_n{n}',F=F,T=T,signal=sig,window=w,entry=e,exit=0.0,n_pairs=n,max_uses=mu,criterion=c,params=P)
                o,net=L.evaluate(D,d); rows.append(dict(n=n,crit=c,sharpe=o['sharpe'],gross=o['gross_ret'],ret=o['ann_ret'],turn=o['turnover']))
        df=pd.DataFrame(rows)
        x=df.pivot(index='crit',columns='n',values='sharpe').round(2); x['gross50']=df[df.n==50].set_index('crit').gross.round(3)
        print(x.sort_values(50,ascending=False).to_string()); print('turnover', df.groupby('n').turn.mean().round(0).to_dict(), flush=True)
