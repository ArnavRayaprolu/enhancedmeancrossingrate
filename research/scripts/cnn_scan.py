"""CNN trading policy vs z thresholds, full walk-forward 2016-2026. One network per fold
(trained on formation data only), shared by every selection criterion."""
import sys, time, warnings; sys.path.insert(0,'.'); warnings.filterwarnings('ignore')
from research import emcr_lab as L, markets as M
import pandas as pd
mk=sys.argv[1]; same=None if sys.argv[2]=='all' else sys.argv[2]
OUT=f'research/cnn_scan_{mk}_{sys.argv[2]}.csv'
crits=['Correlation','Distance','Cointegration','HalfLife','EMCR','EMCR1','EMCRk','RawMean2pc','RawMA20','VW10_40c','VW5_20c',
       'Correlation|EMCR1','Correlation|EMCRkcyc','Correlation|VW10_40c','Correlation|RawMean2pc','Correlation|VW5_20c']
if mk=='large_hourly':
    D=L.daily_data(); cfg=dict(top_k=250,borrow=0.005,params=dict(half_bps=2.0,roll_windows=(5,)))
else:
    D,cfg=M.market(mk); cfg['params']=dict(cfg['params'],roll_windows=(5,))
cfg['params']=dict(cfg['params'],same=same)
rows=[]; t0=time.time()
for F,T in ((252,63),):
    for c in crits:
        for n,mu in ((20,1),(50,2)):
            des=L.Design(label=f'CNN_{mk}_{sys.argv[2]}_F{F}T{T}_n{n}',F=F,T=T,signal='cnn',entry=2.0,exit=0.0,top_k=cfg['top_k'],
                         n_pairs=n,max_uses=mu,criterion=c,borrow_pa=cfg['borrow'],params=cfg['params'])
            o,net=L.evaluate(D,des,part='all')
            net.to_frame('net').to_parquet(f'research/cnn_nets/{mk}_{sys.argv[2]}_{c.replace("|","_I_")}_n{n}.parquet')
            rows.append(dict(market=mk,same=sys.argv[2],F=F,T=T,n=n,crit=c,sharpe=o['sharpe'],ret=o['ann_ret'],vol=o['ann_vol'],gross=o['gross_ret'],turn=o['turnover']))
            print(f'{c} n{n}: SR={o["sharpe"]:.2f} ret={o["ann_ret"]:.2%} [{time.time()-t0:.0f}s]',flush=True)
        pd.DataFrame(rows).to_csv(OUT,index=False)
print('ALL DONE',flush=True)
