"""Innovation scan, full walk-forward 2016-2026: Kalman EMCR, volume-weighted MA-crossover
EMCR, variance ratio and hybrids, with static / Kalman / VWMA trading signals."""
import sys, time, warnings; sys.path.insert(0,'.'); warnings.filterwarnings('ignore')
from research import emcr_lab as L, markets as M
import pandas as pd
mk=sys.argv[1]; same=None if len(sys.argv)<3 or sys.argv[2]=='all' else sys.argv[2]
OUT=f'research/innov_scan_{mk}_{same or "all"}.csv'
crits=['Correlation','Distance','Cointegration','HalfLife','VR','EMCR','EMCRk','EMCRkcyc','EMCRma','EMCRmacyc','EMCRvw','EMCRvwcyc',
       'EMCRkcyc+Correlation','EMCRvwcyc+Cointegration','EMCRvwcyc+Correlation','Correlation|EMCRkcyc','Correlation|EMCRvwcyc',
       'EMCRk+VR','EMCRvwcyc+VR','Cointegration|EMCRvwcyc']
designs=[dict(F=252,T=63,signal='static',entry=2.0,exit=0.0),
         dict(F=126,T=63,signal='static',entry=2.5,exit=0.5),
         dict(F=252,T=63,signal='kalman',entry=2.0,exit=0.0),
         dict(F=252,T=63,signal='kalman',entry=1.5,exit=0.0),
         dict(F=126,T=21,signal='vwma',entry=1.0,exit=0.25)]
if mk=='large_hourly':
    D=L.daily_data(); cfg=dict(top_k=250,borrow=0.005,params=dict(half_bps=2.0,roll_windows=(5,)))
else:
    D,cfg=M.market(mk); cfg['params']=dict(cfg['params'],roll_windows=(5,))
cfg['params']=dict(cfg['params'],same=same)
rows=[]; t0=time.time()
for dz in designs:
    for c in crits:
        for n,mu in ((20,1),(50,2)):
            des=L.Design(label=f'IN_{mk}_{same}_F{dz["F"]}T{dz["T"]}_{dz["signal"]}_e{dz["entry"]}x{dz["exit"]}_n{n}',
                         top_k=cfg['top_k'],n_pairs=n,max_uses=mu,criterion=c,borrow_pa=cfg['borrow'],params=cfg['params'],**dz)
            o,net=L.evaluate(D,des,part='all')
            rows.append(dict(market=mk,same=same,**dz,n=n,crit=c,sharpe=o['sharpe'],ret=o['ann_ret'],vol=o['ann_vol'],gross=o['gross_ret'],turn=o['turnover']))
    print(f'{mk} {same} {dz} done [{time.time()-t0:.0f}s]', flush=True)
    pd.DataFrame(rows).to_csv(OUT,index=False)
print('ALL DONE', flush=True)
