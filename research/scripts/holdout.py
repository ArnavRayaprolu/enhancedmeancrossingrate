import sys, warnings; sys.path.insert(0,'.'); warnings.filterwarnings('ignore')
from research import emcr_lab as L
from mcr import inference as I
import pandas as pd, numpy as np
D=L.daily_data()
def des(same,F,T,e,x,n,mu,c): return L.Design(label=f'HOLDOUT_{same}_F{F}T{T}_e{e}x{x}_n{n}_{c}',F=F,T=T,signal='static',entry=e,exit=x,n_pairs=n,max_uses=mu,criterion=c,params=dict(roll_windows=(5,),half_bps=2.0,same=same))
cands={'A: best dev EMCR hybrid (Correlation|EMCR1, subind 6m/3m)':('subind',126,63,2.5,0.5,50,2,'Correlation|EMCR1'),
       'B: same design, Correlation':('subind',126,63,2.5,0.5,50,2,'Correlation'),
       'C: best dev Correlation (subind 12m/6m)':('subind',252,126,2.5,0.5,50,2,'Correlation'),
       'D: same design as C, Correlation|EMCR1':('subind',252,126,2.5,0.5,50,2,'Correlation|EMCR1')}
nets={}
for k,g in cands.items():
    for part in ('dev','holdout'):
        o,net=L.evaluate(D,des(*g),part=part); nets[(k,part)]=net
        print(f'{part:8s} {k:60s} SR={o["sharpe"]:.2f} ret={o["ann_ret"]:.2%} vol={o["ann_vol"]:.2%} gross={o["gross_ret"]:.2%} days={o["days"]}', flush=True)
for a,b in (('A: best dev EMCR hybrid (Correlation|EMCR1, subind 6m/3m)','B: same design, Correlation'),('D: same design as C, Correlation|EMCR1','C: best dev Correlation (subind 12m/6m)')):
    x,y=nets[(a,'holdout')].align(nets[(b,'holdout')],join='inner')
    t=I.diebold_mariano(x.to_numpy(),y.to_numpy(),'nw'); print('holdout DM',a[:2],'vs',b[:2],round(t['dm'],2),'p=',round(t['p'],3))
