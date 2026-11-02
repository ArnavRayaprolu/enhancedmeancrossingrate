"""Trading engine (reviewer comment R4) and capital allocation (Eq. 8).

Timing. Bar t is the hour ending at its close. The signal z_t uses closes up to
and including t. A decision taken at the close of t is executed at the close of
t + lag (lag = 0: same close, as in the paper) and the position earns the
returns of the following bars.

Position rules, 'revised' engine (applied identically to every criterion):

  state  = 0 (flat), +1 (long the ratio: buy A, sell B), -1 (short the ratio)
  flat   : z_t <= -entry -> +1 ;  z_t >= +entry -> -1 ;  otherwise stay flat
  long   : z_t >= +entry           -> -1   (opposite extreme: close and reverse)
           z_t >= -exit            ->  0   (signal back in, or through, the exit band)
           otherwise hold
  short  : symmetric
  z_t missing (warm-up, halted leg): hold the current state, open nothing
  a new signal in the direction already held is ignored (no pyramiding)
  every test window starts flat, and all positions are liquidated at the
  close of the last test bar
  the entry rule has no upper bound: |z| >= entry. The 1.00-1.25 band drawn in
  the original Figure 3 was illustrative only.

'original' engine = the submitted notebook: exit only when |z_t| <= exit (so a
jump from one extreme to the other keeps the old position), no reversal, and
the state machine runs over formation + test bars so a position can be carried
into the test window; no forced liquidation at the window end.
"""
from __future__ import annotations

import numpy as np


def states(z: np.ndarray, entry: float, exit_: float, engine: str = "revised",
           start: int = 0) -> np.ndarray:
    """Decision states for rows start..T-1 of z (T, n).  Row k of the output is
    the state decided at the close of bar start + k."""
    T, n = z.shape
    st = np.zeros(n)
    out = np.zeros((T - start, n))
    first = 0 if engine == "original" else start
    for t in range(first, T):
        zt = z[t]
        ok = np.isfinite(zt)
        if engine == "original":
            open_long = ok & (st == 0) & (zt <= -entry)
            open_short = ok & (st == 0) & (zt >= entry)
            close = ok & (st != 0) & (np.abs(zt) <= exit_)
            st = np.where(open_long, 1.0, np.where(open_short, -1.0, np.where(close, 0.0, st)))
        else:
            new = st.copy()
            flat, lng, sht = st == 0, st > 0, st < 0
            new[ok & flat & (zt <= -entry)] = 1.0
            new[ok & flat & (zt >= entry)] = -1.0
            new[ok & lng & (zt >= -exit_)] = 0.0
            new[ok & lng & (zt >= entry)] = -1.0
            new[ok & sht & (zt <= exit_)] = 0.0
            new[ok & sht & (zt <= -entry)] = 1.0
            st = new
        if t >= start:
            out[t - start] = st
    return out


def completed_cycles(z: np.ndarray, entry: float, exit_: float) -> np.ndarray:
    """Number of round trips (open -> close or reversal) the revised engine
    completes on z; a diagnostic of 'effective trading cycles' (R1)."""
    s = states(z, entry, exit_, "revised")
    prev = np.vstack([np.zeros((1, s.shape[1])), s[:-1]])
    return ((prev != 0) & (s != prev)).sum(0)


# --------------------------------------------------------------------------- #
# portfolio accounting
# --------------------------------------------------------------------------- #
def pair_weights(pos: np.ndarray, phi: np.ndarray, rule: str, cap: float) -> np.ndarray:
    """Capital weight w_i,t of each selected pair (fraction of NAV, gross).

    phi_i  = the pair's selection score: its percentile rank under the
             criterion that selected it (hybrids: the average of the two
             percentile ranks), in (0, 1]; not renormalised before weighting.
    active (Eq. 8)  w_i,t = phi_i 1{pos_i,t != 0} / sum_j phi_j 1{pos_j,t != 0}
    capped          min(active weight, cap)  -- no pair can take more than `cap`
    rank            phi_i / sum_j phi_j  over all selected pairs, fixed in the window
    equal           1 / N
    Inactive pairs hold no capital under any rule; idle capital is uninvested.
    """
    act = pos != 0
    if rule in ("active", "capped"):
        s = np.where(act, phi[None, :], 0.0)
        tot = s.sum(1, keepdims=True)
        w = np.divide(s, tot, out=np.zeros_like(s), where=tot > 0)
        if rule == "capped":
            w = np.minimum(w, cap)
        return w
    if rule == "rank":
        return np.where(act, (phi / phi.sum())[None, :], 0.0)
    if rule == "equal":
        return np.where(act, 1.0 / len(phi), 0.0)
    raise ValueError(rule)


def run_book(pos: np.ndarray, w: np.ndarray, ra: np.ndarray, rb: np.ndarray,
             hedge_a: np.ndarray, cost_a: np.ndarray, cost_b: np.ndarray,
             borrow_bar: float, impact: tuple | None = None) -> dict:
    """Simulate the book over the test window.

    pos, w   (K, n) executed state and pair weight, row k = held over bar k+1
             of the window and decided/executed at the close of bar k; row 0
             is executed at the close of the last formation bar.
    ra, rb   (K, n) simple returns of leg A / leg B over test bars 0..K-1
    hedge_a  (n,)  share of the pair's gross weight on leg A (0.5 for the
             equal-dollar ratio trade; 1/(1+beta) for the hedged spread)
    cost_a/b (n,)  one-way cost per dollar traded on leg A / leg B
    impact   optional (sigma_a, sigma_b, dvol_a, dvol_b, coef, aum): extra
             one-way cost coef * sigma * sqrt(trade $ / bar dollar volume)

    Holdings are in dollars per dollar of NAV. Between decisions a pair's
    holdings drift with prices; the book trades a pair only when its target
    (state or weight) changes, so continuous rebalancing is never assumed for
    free and every resizing is charged (R5). Turnover counts both legs.
    Returns per-bar gross, cost, borrow, net, turnover and long/short gross.
    """
    K, n = pos.shape
    ha_t = pos * w * hedge_a[None, :]                  # target dollar holdings
    hb_t = -pos * w * (1 - hedge_a)[None, :]
    ha = np.zeros(n); hb = np.zeros(n)
    prev_key = np.zeros((n, 2))
    gross = np.zeros(K); cost = np.zeros(K); borrow = np.zeros(K); turn = np.zeros(K)
    expo = np.zeros(K)

    def trade_cost(da, db):
        c = np.abs(da) * cost_a + np.abs(db) * cost_b
        if impact is not None:
            sa, sb, va, vb, coef, aum = impact
            c = c + coef * (sa * np.abs(da) * np.sqrt(np.abs(da) * aum / va)
                            + sb * np.abs(db) * np.sqrt(np.abs(db) * aum / vb))
        return float(np.nansum(c))

    for k in range(K):
        # trade at the close of bar k-1 (k = 0: close of the last formation bar)
        key = np.stack([pos[k], w[k]], 1)
        change = np.any(key != prev_key, axis=1)
        da = np.where(change, ha_t[k] - ha, 0.0)
        db = np.where(change, hb_t[k] - hb, 0.0)
        ha = ha + da; hb = hb + db
        prev_key = key
        cost[k] += trade_cost(da, db)
        turn[k] += float(np.abs(da).sum() + np.abs(db).sum())
        expo[k] = float(np.abs(ha).sum() + np.abs(hb).sum())
        # earn bar k
        g = float((ha * ra[k]).sum() + (hb * rb[k]).sum())
        borrow[k] = borrow_bar * float(np.abs(np.minimum(ha, 0)).sum()
                                       + np.abs(np.minimum(hb, 0)).sum())
        gross[k] = g
        ha = ha * (1 + ra[k]) / (1 + g)
        hb = hb * (1 + rb[k]) / (1 + g)
    # liquidate at the close of the last bar
    cost[K - 1] += trade_cost(-ha, -hb)
    turn[K - 1] += float(np.abs(ha).sum() + np.abs(hb).sum())
    net = gross - cost - borrow
    return dict(gross=gross, cost=cost, borrow=borrow, net=net, turn=turn, exposure=expo)
