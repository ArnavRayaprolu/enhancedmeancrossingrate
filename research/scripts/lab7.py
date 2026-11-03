import sys, warnings; sys.path.insert(0,'.'); warnings.filterwarnings('ignore')
from research import emcr_lab as L
import pandas as pd, numpy as np
D=L.daily_data()
def des(same,F,T,e,x,n,mu,c): return L.Design(label=f'E7_{same}_F{F}T{T}_e{e}x{x}_n{n}_{c}',F=F,T=T,signal='static',entry=e,exit=x,n_pairs=n,max_uses=mu,criterion=c,params=dict(roll_windows=(5,),half_bps=2.0,same=same))
legs=[('subind',126,63,2.5,0.5,50,2),('subind',252,126,2.5,0.5,100,3),('sector',252,126,2.5,0.5,50,2),('sector',126,63,2.5,0.5,50,2)]
def sr(x): return x.mean()/x.std()*np.sqrt(252)
for c in ('Correlation|EMCR1','Correlation','EMCR2','Distance','Cointegration'):
    s=[]
    for g in legs:
        o,net=L.evaluate(D,des(*g,c)); s.append(net.rename(str(g)))
    M=pd.concat(s,axis=1).fillna(0)
    ens=M.mean(axis=1)
    print(f'{c:20s} legs: {[round(sr(M[k]),2) for k in M]}  ensemble SR={sr(ens):.2f} ret={ens.mean()*252:.3%} vol={ens.std()*np.sqrt(252):.3%}  corr(legs) mean={M.corr().values[np.triu_indices(4,1)].mean():.2f}', flush=True)
