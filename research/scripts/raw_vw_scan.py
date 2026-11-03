"""Raw price-ratio EMCR and volume-weighted MA-combination EMCR, threshold engine,
full walk-forward 2016-2026."""
import sys, time, warnings; sys.path.insert(0,'.'); warnings.filterwarnings('ignore')
from research import emcr_lab as L, markets as M
import pandas as pd
mk=sys.argv[1]; same=None if sys.argv[2]=='all' else sys.argv[2]
OUT=f'research/raw_vw_scan_{mk}_{sys.argv[2]}.csv'
new=['RawMean','RawMean2pc','RawMean5pc','RawMA10','RawMA20','RawMA60','RawMA10c','RawMA20c','RawMA60c',
     'VW3_10','VW5_20','VW10_40','VW20_60','VW3_10c','VW5_20c','VW10_40c','VW20_60c']
classic=['Correlation','Distance','Cointegration','HalfLife']
hyb=[f'Correlation|{c}' for c in ('RawMean2pc','RawMA20c','VW5_20c','VW10_40c')]+[f'{c}+Correlation' for c in ('RawMean2pc','VW10_40c')]
crits=classic+new+hyb
designs=[dict(F=126,T=63,signal='static',entry=2.5,exit=0.5),dict(F=252,T=63,signal='static',entry=2.0,exit=0.0),
         dict(F=126,T=21,signal='vwma',entry=1.0,exit=0.25)]
if mk=='large_hourly':
    D=L.daily_data(); cfg=dict(top_k=250,borrow=0.005,params=dict(half_bps=2.0,roll_windows=(5,)))
else:
    D,cfg=M.market(mk); cfg['params']=dict(cfg['params'],roll_windows=(5,))
cfg['params']=dict(cfg['params'],same=same,kalman=False)
rows=[]; t0=time.time()
for dz in designs:
    for c in crits:
        for n,mu in ((20,1),(50,2)):
            des=L.Design(label=f'RV_{mk}_{sys.argv[2]}_F{dz["F"]}T{dz["T"]}_{dz["signal"]}_n{n}',top_k=cfg['top_k'],n_pairs=n,max_uses=mu,
                         criterion=c,borrow_pa=cfg['borrow'],params=cfg['params'],**dz)
            o,_=L.evaluate(D,des,part='all')
            rows.append(dict(market=mk,same=sys.argv[2],**dz,n=n,crit=c,sharpe=o['sharpe'],ret=o['ann_ret'],vol=o['ann_vol'],turn=o['turnover']))
    print(f'{mk} {sys.argv[2]} {dz} done [{time.time()-t0:.0f}s]',flush=True)
    pd.DataFrame(rows).to_csv(OUT,index=False)
print('ALL DONE',flush=True)
