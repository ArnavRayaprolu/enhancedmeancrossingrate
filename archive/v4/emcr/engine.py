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


def run_states(z: np.ndarray, entry: float, exit_: float, stop: float | None = None,
               max_hold: int | None = None, allow: np.ndarray | None = None,
               confirm: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised state machine over the columns of z (T x P).

    Optional risk exits (off by default):
      stop      close if the divergence widens to |z| >= stop against the position
      max_hold  close after max_hold bars in the trade
    After a risk exit the pair is not re-entered until z has come back inside
    the entry band (|z| < entry), so a stopped-out trade is not re-opened at once.
      confirm   confirmation entry: open only when z crosses back INSIDE the band
                after having been outside it (z_{t-1} <= -entry < z_t opens long;
                z_{t-1} >= entry > z_t opens short), i.e. once reversion has begun

    Returns
    -------
    states : (T, P) state after processing bar t
    n_closed : (P,) number of round trips closed by reversion to the exit level
    """
    T, P = z.shape
    s = np.zeros(P); held = np.zeros(P); armed = np.ones(P, dtype=bool)
    out = np.zeros((T, P))
    closed = np.zeros(P)
    zprev = np.full(P, np.nan)
    for t in range(T):
        zt = z[t]
        ok = np.isfinite(zt)
        held = np.where(s != 0, held + 1, 0)          # bars held, incl. this one
        rev = ok & (((s > 0) & (zt >= -exit_)) | ((s < 0) & (zt <= exit_)))
        risk = np.zeros(P, dtype=bool)
        if stop is not None:
            risk |= ok & (((s > 0) & (zt <= -stop)) | ((s < 0) & (zt >= stop)))
        if max_hold is not None:
            risk |= (s != 0) & (held >= max_hold)
        risk &= ~rev
        closed += rev
        s = np.where(rev | risk, 0.0, s)
        armed = np.where(risk, False, armed)
        armed = np.where(ok & (np.abs(zt) < entry), True, armed)
        flat = ok & (s == 0) & armed
        if allow is not None:
            flat &= allow[t]
        if confirm:
            s = np.where(flat & (zprev <= -entry) & (zt > -entry) & (zt < -exit_), 1.0, s)
            s = np.where(flat & (zprev >= entry) & (zt < entry) & (zt > exit_), -1.0, s)
        else:
            s = np.where(flat & (zt <= -entry), 1.0, s)
            s = np.where(flat & (zt >= entry), -1.0, s)
        zprev = np.where(ok, zt, zprev)
        out[t] = s
    return out, closed


def completed_cycle_rate(z: np.ndarray, entry: float, exit_: float, **kw) -> np.ndarray:
    """CCR: completed entry-to-exit cycles per valid bar in the formation window.

    This is the empirical N(T)/T of the paper's Eq. (14): the number of round
    trips the engine itself would have completed in-sample.  Unlike MCR it
    ignores small oscillations around zero that never reach the entry band.
    """
    _, closed = run_states(z, entry, exit_, **kw)
    n = np.isfinite(z).sum(axis=0)
    out = np.full(z.shape[1], np.nan)
    ok = n > 0
    out[ok] = closed[ok] / n[ok]
    return out


def positions_for_window(z: np.ndarray, start: int, end: int, entry: float,
                         exit_: float, **kw) -> np.ndarray:
    """Position held over each return bar t in [start, end).

    pos[t] = state after processing z[t-1]; decisions start at bar `start`
    (so the first bar is flat) and no decision is taken at the last bar
    (its position is liquidated at the close of the window)."""
    if kw.get("allow") is not None:
        kw = dict(kw, allow=kw["allow"][start:end - 1])
    st, _ = run_states(z[start:end - 1], entry, exit_, **kw)
    pos = np.zeros((end - start, z.shape[1]))
    pos[1:] = st
    return pos


# --------------------------------------------------------------------------- #
# portfolio accounting
# --------------------------------------------------------------------------- #
def portfolio(pos: np.ndarray, rets_a: np.ndarray, rets_b: np.ndarray,
              a: np.ndarray, b: np.ndarray, score: np.ndarray, weighting: str,
              borrow_per_bar: float, fund_a: np.ndarray | None = None,
              fund_b: np.ndarray | None = None) -> dict:
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
    elif weighting == "rank":
        # rank-weighted over ALL selected pairs, fixed for the window: a pair's
        # capital does not change when other pairs open or close (no re-splitting)
        w = active * (score / score.sum())[None, :]
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
    if fund_a is not None:
        # perpetual funding settled during the bar: longs pay a positive rate
        borrow = borrow + (h * np.concatenate([fund_a, fund_b], axis=1)).sum(axis=1)
    return dict(gross=gross, turn=turn, borrow=borrow, W=W, h=h,
                exposure=np.abs(W).sum(axis=1))
