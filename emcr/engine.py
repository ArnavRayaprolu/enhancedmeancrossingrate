"""Trading engine: an explicit, fully specified position state machine.

State machine (applied independently to each pair, bar by bar)
---------------------------------------------------------------
state s in {-1, 0, +1};  +1 = long the spread (long A / short B), -1 = short.
Thresholds: entry e > 0, exit x with 0 <= x < e.

At each bar t, given z_t (NaN = no decision, the state is held):
  1. EXIT   if s = +1 and z_t >= -x   (z has reverted to within x of zero, or crossed it)
            if s = -1 and z_t <= +x
  2. ENTER  if s =  0 after step 1:  s = +1 if z_t <= -e;  s = -1 if z_t >= +e
Consequences that answer the "all trajectories" question:
  * a jump straight from one entry band to the other (e.g. z: +1.2 -> -1.1)
    closes the short and opens a long on the same bar (a reversal);
  * while a position is open, further signals in the same direction are ignored
    (no pyramiding, no re-weighting by signal strength);
  * the state decided at the close of bar t is held over bar t+1 and earns bar
    t+1's return (no contemporaneous information);
  * every trading window starts flat and every open position is closed at the
    close of the last bar of the window (forced liquidation, cost charged).
"""
from __future__ import annotations

import numpy as np


def run_states(z: np.ndarray, entry: float, exit_: float) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised state machine over the columns of z (T x P).

    Returns
    -------
    states : (T, P) state after processing bar t
    n_closed : (P,) number of completed round trips (exit events)
    """
    T, P = z.shape
    s = np.zeros(P)
    out = np.zeros((T, P))
    closed = np.zeros(P)
    for t in range(T):
        zt = z[t]
        ok = np.isfinite(zt)
        ex = ok & (((s > 0) & (zt >= -exit_)) | ((s < 0) & (zt <= exit_)))
        closed += ex
        s = np.where(ex, 0.0, s)
        flat = ok & (s == 0)
        s = np.where(flat & (zt <= -entry), 1.0, s)
        s = np.where(flat & (zt >= entry), -1.0, s)
        out[t] = s
    return out, closed


def completed_cycle_rate(z: np.ndarray, entry: float, exit_: float) -> np.ndarray:
    """CCR: completed entry-to-exit cycles per valid bar in the formation window.

    This is the empirical N(T)/T of the paper's Eq. (14): the number of round
    trips the engine itself would have completed in-sample.  Unlike SCR it
    ignores small oscillations around zero that never reach the entry band.
    """
    _, closed = run_states(z, entry, exit_)
    n = np.isfinite(z).sum(axis=0)
    out = np.full(z.shape[1], np.nan)
    ok = n > 0
    out[ok] = closed[ok] / n[ok]
    return out


def positions_for_window(z: np.ndarray, start: int, end: int, entry: float,
                         exit_: float) -> np.ndarray:
    """Position held over each return bar t in [start, end).

    pos[t] = state after processing z[t-1]; decisions start at bar `start`
    (so the first bar is flat) and no decision is taken at the last bar
    (its position is liquidated at the close of the window)."""
    st, _ = run_states(z[start:end - 1], entry, exit_)
    pos = np.zeros((end - start, z.shape[1]))
    pos[1:] = st
    return pos


# --------------------------------------------------------------------------- #
# portfolio accounting
# --------------------------------------------------------------------------- #
def portfolio(pos: np.ndarray, rets_a: np.ndarray, rets_b: np.ndarray,
              a: np.ndarray, b: np.ndarray, score: np.ndarray, weighting: str,
              borrow_per_bar: float) -> dict:
    """Account a portfolio of N pairs over one trading window.

    pos      (T, N)  position (+1/0/-1) held over each bar
    rets_*   (T, N)  simple returns of leg A / leg B over each bar
    a, b     (N,)    leg dollar weights for a +1 unit position, |a| + |b| = 1
    score    (N,)    formation-window selection score Phi_i (> 0)
    weighting "active": W = pos * Phi * 1{active} / sum_j Phi_j 1{active}   (paper Eq. 8)
              "equal" : W = pos / N  (committed capital, idle when flat)

    Leg holdings h (T, 2N) are the dollar weights held over each bar.  Between
    bars, holdings drift with prices; turnover is the amount traded to move the
    drifted book back to the next target book, so re-sizing live positions when
    the active set changes AND re-balancing price drift are both charged.  The
    book is liquidated at the close of the final bar.
    """
    T, N = pos.shape
    active = pos != 0
    if weighting == "active":
        sc = np.where(active, score[None, :], 0.0)
        tot = sc.sum(axis=1, keepdims=True)
        w = np.divide(sc, tot, out=np.zeros_like(sc), where=tot > 0)
    elif weighting == "equal":
        w = active / N
    else:
        raise ValueError(weighting)
    W = w * pos                                            # signed pair weights
    h = np.concatenate([W * a[None, :], W * b[None, :]], axis=1)
    r = np.concatenate([rets_a, rets_b], axis=1)
    pnl = h * r
    gross = pnl.sum(axis=1)
    drifted = (h + pnl) / (1.0 + gross)[:, None]           # book at the close of bar t
    prev = np.vstack([np.zeros((1, 2 * N)), drifted[:-1]])
    turn = np.abs(h - prev).sum(axis=1)
    turn[-1] += np.abs(drifted[-1]).sum()                  # final liquidation
    short = np.clip(-h, 0, None).sum(axis=1)
    borrow = short * borrow_per_bar
    return dict(gross=gross, turn=turn, borrow=borrow, W=W, h=h,
                exposure=np.abs(W).sum(axis=1))
