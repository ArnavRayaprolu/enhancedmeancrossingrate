"""Integrity checks: look-ahead, accounting, trade direction, statistics."""
import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.stattools import coint

from emcr.criteria import engle_granger_pvalue, pct_rank, select
from emcr.engine import completed_cycle_rate, portfolio, positions_for_window, run_states
from emcr.signals import (half_life, rolling_mean, rolling_std, signal_crossing_rate,
                          spread, zscore)

rng = np.random.default_rng(0)


def _prices(T=900, P=6):
    return 100 * np.exp(np.cumsum(rng.normal(0, 0.01, (T, P)), axis=0))


# ---------------------------------------------------------------- signals ---
def test_rolling_matches_pandas():
    x = _prices()
    df = pd.DataFrame(x)
    assert np.allclose(rolling_mean(x, 35), df.rolling(35).mean(), equal_nan=True)
    assert np.allclose(rolling_std(x, 140), df.rolling(140).std(), equal_nan=True)


@pytest.mark.parametrize("sig", ["MAD", "LEVEL", "HEDGED"])
def test_no_lookahead_in_z(sig):
    """Perturbing prices after a cut-off must not change z before it."""
    p = _prices()
    q = p.copy(); cut = 600
    q[cut:] *= rng.uniform(0.5, 1.5, q[cut:].shape)
    beta = np.ones(3)
    z1 = zscore(spread(p[:, :3], p[:, 3:], sig, beta), sig, 35, 140)
    z2 = zscore(spread(q[:, :3], q[:, 3:], sig, beta), sig, 35, 140)
    assert np.allclose(z1[:cut], z2[:cut], equal_nan=True)
    pos1 = positions_for_window(z1, 500, 800, 1.0, 0.25)
    pos2 = positions_for_window(z2, 500, 800, 1.0, 0.25)
    # position over bar t depends on z[t-1]: identical up to and including bar `cut`
    assert np.array_equal(pos1[: cut - 500 + 1], pos2[: cut - 500 + 1])


def test_engle_granger_matches_statsmodels():
    p = np.log(_prices(630, 8))
    p[:, 1] = p[:, 0] + rng.normal(0, 0.01, 630)          # one cointegrated pair
    ia, ib = np.array([0, 2, 4, 6]), np.array([1, 3, 5, 7])
    ours = engle_granger_pvalue(p[:, ia], p[:, ib])
    ref = [coint(p[:, i], p[:, j], trend="c", maxlag=1, autolag=None)[1]
           for i, j in zip(ia, ib)]
    assert np.allclose(ours, ref, atol=1e-8)


def test_scr_definition():
    z = np.array([[1.0], [0.0], [-1.0], [np.nan], [-2.0], [0.0], [0.0], [3.0]])
    # valid (non-zero, finite): 1, -1, -2, 3 -> changes: 1->-1, -2->3 = 2 over 3 steps
    assert signal_crossing_rate(z)[0] == pytest.approx(2 / 3)
    assert signal_crossing_rate(np.ones((50, 1)))[0] == 0.0
    assert signal_crossing_rate(np.array([[1.0, -1.0] * 25]).T)[0] == 1.0


def test_scr_arcsine_law():
    """For a Gaussian AR(1) the sign-change probability is arccos(phi)/pi."""
    for phi in (0.8, 0.9, 0.97):
        e = rng.normal(size=200_000); x = np.empty_like(e); x[0] = 0
        for t in range(1, len(e)):
            x[t] = phi * x[t - 1] + e[t]
        assert signal_crossing_rate(x[:, None])[0] == pytest.approx(np.arccos(phi) / np.pi,
                                                                    abs=0.003)


def test_half_life_ar1():
    phi = 0.95
    e = rng.normal(size=50_000); x = np.zeros_like(e)
    for t in range(1, len(e)):
        x[t] = phi * x[t - 1] + e[t]
    assert half_life(x[:, None])[0] == pytest.approx(np.log(2) / -np.log(phi), rel=0.1)


# ----------------------------------------------------------------- engine ---
def test_state_machine_paths():
    z = np.array([0.0, -1.2, -0.8, -0.2, 0.5, 1.3, 0.9, -1.4, -1.1, 0.1, np.nan, 0.0])[:, None]
    st, closed = run_states(z, 1.0, 0.25)
    #            0   -1.2 -0.8 -0.2 0.5 1.3  0.9 -1.4 -1.1 0.1 nan 0.0
    expected = [0, 1, 1, 0, 0, -1, -1, 1, 1, 0, 0, 0]
    assert st[:, 0].tolist() == expected
    # exits: at -0.2, at -1.4 (reversal through zero), at 0.1  -> 3 closed
    assert closed[0] == 3


def test_ccr_counts_band_to_mean_cycles():
    # small wiggles around zero never reach the band: SCR high, CCR zero
    z = (0.3 * np.sin(np.arange(400) / 2.0))[:, None]
    assert signal_crossing_rate(z)[0] > 0.15
    assert completed_cycle_rate(z, 1.0, 0.25)[0] == 0.0
    big = (1.5 * np.sin(np.arange(400) / 10.0))[:, None]
    assert completed_cycle_rate(big, 1.0, 0.25)[0] > 0


def test_window_starts_flat_and_ends_flat():
    z = np.full((50, 1), -2.0)
    pos = positions_for_window(z, 10, 40, 1.0, 0.25)
    assert pos[0, 0] == 0 and (pos[1:, 0] == 1).all()


def test_direction_profits_on_reversion():
    T = 300
    pb = np.full(T, 100.0); pa = np.full(T, 100.0)
    pa[150:160] = 90.0                                  # A cheap -> buy A / short B
    z = zscore((pa / pb)[:, None], "LEVEL", 1, 30)
    pos = positions_for_window(z, 140, 200, 2.0, 0.5)
    ra = np.r_[0, pa[1:] / pa[:-1] - 1][140:200, None]
    rb = np.zeros_like(ra)
    res = portfolio(pos, ra, rb, np.array([0.5]), np.array([-0.5]), np.array([1.0]),
                    "equal", 0.0)
    assert pos.max() == 1 and res["gross"].sum() > 0


def test_cost_accounting_round_trip():
    pos = np.array([[0, 1, 1, 1, 0]], dtype=float).T
    z0 = np.zeros((5, 1))
    res = portfolio(pos, z0, z0, np.array([0.5]), np.array([-0.5]), np.array([1.0]),
                    "equal", 0.0)
    # open 0.5 + 0.5, close 0.5 + 0.5 -> 2 units of notional per round trip
    assert res["turn"].sum() == pytest.approx(2.0)


def test_forced_liquidation_charged():
    pos = np.array([[0, 1, 1]], dtype=float).T
    z0 = np.zeros((3, 1))
    res = portfolio(pos, z0, z0, np.array([0.5]), np.array([-0.5]), np.array([1.0]),
                    "equal", 0.0)
    assert res["turn"].sum() == pytest.approx(2.0)


def test_active_weights_unlevered_and_neutral():
    T, N = 200, 5
    pos = rng.choice([-1.0, 0.0, 1.0], size=(T, N))
    r = rng.normal(0, 0.01, (T, N))
    res = portfolio(pos, r, -r, np.full(N, 0.5), np.full(N, -0.5), rng.uniform(0.5, 1, N),
                    "active", 0.0)
    assert res["exposure"].max() <= 1 + 1e-12
    assert np.allclose(res["h"].sum(1), 0.0)            # dollar neutral book


def test_drift_rebalancing_is_charged():
    pos = np.ones((3, 1)); ra = np.array([[0.0], [0.10], [0.0]]); rb = np.zeros((3, 1))
    res = portfolio(pos, ra, rb, np.array([0.5]), np.array([-0.5]), np.array([1.0]),
                    "equal", 0.0)
    # bar 0 opens (1.0); after bar 1 the A leg has grown and must be trimmed
    assert res["turn"][2] > 0


# ------------------------------------------------------------- selection ---
def test_rank_orientation_and_missing():
    x = np.array([3.0, 1.0, np.nan, 2.0])
    assert pct_rank(x, True).tolist() == [1.0, 1 / 3, 0.0, 2 / 3]
    assert pct_rank(x, False)[1] == 1.0


def test_select_no_reuse():
    score = np.array([0.9, 0.8, 0.7, 0.6])
    ia = np.array([0, 0, 1, 2]); ib = np.array([1, 2, 3, 3])
    assert select(score, ia, ib, 3).tolist() == [0, 3]


def test_stop_loss_and_rearm():
    z = np.array([-2.1, -2.8, -3.1, -2.5, -2.2, -1.5, -2.2, -0.5, 0.1])[:, None]
    st, closed = run_states(z, 2.0, 0.0, stop=3.0)
    # stopped out at -3.1, not re-entered at -2.5/-2.2 (still outside the band),
    # re-armed at -1.5, re-entered at -2.2, closed at the mean (0.1)
    assert st[:, 0].tolist() == [1, 1, 0, 0, 0, 0, 1, 1, 0]
    assert closed[0] == 1


def test_max_hold():
    z = np.full((10, 1), -2.5)
    st, _ = run_states(z, 2.0, 0.0, max_hold=3)
    assert st[:, 0].tolist()[:5] == [1, 1, 1, 0, 0]


def test_funding_sign():
    pos = np.ones((2, 1)); r0 = np.zeros((2, 1))
    fa = np.array([[0.001], [0.0]]); fb = np.zeros((2, 1))
    res = portfolio(pos, r0, r0, np.array([0.5]), np.array([-0.5]), np.array([1.0]),
                    "equal", 0.0, fa, fb)
    assert res["borrow"][0] == pytest.approx(0.5 * 0.001)   # long A pays funding


def test_magnitude_filtered_crossings():
    # ratio deviation d (fraction): tiny wiggles around the mean, then two real
    # excursions of 3% that come back through the mean
    d = np.r_[0.002, -0.002, 0.001, -0.001, 0.03, 0.01, -0.001, -0.03, -0.01, 0.002][:, None]
    assert signal_crossing_rate(d)[0] > 0.5                    # plain count: many
    assert completed_cycle_rate(d, 0.02, 0.0)[0] == pytest.approx(2 / 10)  # filtered: 2


def test_confirmation_entry():
    # threshold entry would open at -2.3; confirmation waits until z turns back inside
    z = np.array([-1.0, -2.3, -2.8, -1.7, -1.2, -0.3, 0.2])[:, None]
    st, closed = run_states(z, 2.0, 0.5, confirm=True)
    assert st[:, 0].tolist() == [0, 0, 0, 1, 1, 0, 0]
    assert closed[0] == 1
