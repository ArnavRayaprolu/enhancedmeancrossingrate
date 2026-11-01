import sys, time
sys.path.insert(0, '.')
from emcr.explore import Spec, evaluate
crit = ["Distance","Correlation","Cointegration","HalfLife","MCR","CCR","MCR+Coint","CCR+Coint"]
for w in ["active","equal"]:
    for c in crit:
        m = evaluate("perp_1h", Spec(criterion=c, signal="STATIC", weighting=w), note="static mean (Gatev) signal")
        print(f"{w:6s} {c:12s} sharpe={m['sharpe']:+.2f} ret={m['total_return']:+.2f} gross={m['gross_pa']:+.2f} turn={m['turnover_pa']:.0f} dd={m['max_dd']:+.2f}", flush=True)
