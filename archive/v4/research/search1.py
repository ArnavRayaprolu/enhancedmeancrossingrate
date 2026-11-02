import sys, time
sys.path.insert(0, '.')
from emcr.explore import Spec, evaluate
crit = ["Distance","Correlation","Cointegration","HalfLife","MCR","CCR","MCR+Coint","CCR+Coint","PCR","MCRxAmp","FormationReturn"]
rows=[]
for w in ["active","equal"]:
    for c in crit:
        t=time.time(); m = evaluate("perp_1h", Spec(criterion=c, weighting=w), note="baseline")
        print(f"{w:6s} {c:16s} sharpe={m['sharpe']:+.2f} ret={m['total_return']:+.2f} gross={m['gross_pa']:+.2f} turn={m['turnover_pa']:.0f} dd={m['max_dd']:+.2f} [{time.time()-t:.0f}s]", flush=True)
