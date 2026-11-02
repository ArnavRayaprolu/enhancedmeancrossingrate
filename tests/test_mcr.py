"""Integrity audit of the revised model (ported from the notebook's section 15,
plus checks for every engine rule stated in mcr/engine.py)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.stattools import coint

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mcr import config as C, engine as E, selection as SEL, signals as S  # noqa: E402

rng = np.random.default_rng(0)


def _pair(T=900):
    b = 50 * np.exp(np.cumsum(rng.normal(0, 0.003, T)))
    a = b * np.exp(0.05 * np.sin(np.arange(T) / 15) + rng.normal(0, 0.002, T))
    return a, b


def test_ma_signal_matches_pandas():
    a, b = _pair()
    r = pd.Series(a / b)
    ref = (r.rolling(35).mean() - r.rolling(140).mean()) / r.rolling(140).std()
    z = S.ma_signal(S.ratio_series(a, b)[:, None], 35, 140)[:, 0]
    assert np.allclose(z[200:], ref.to_numpy()[200:], atol=1e-6)


def test_no_lookahead_in_signal_and_states():
    a, b = _pair(); cut = 600
    a2 = a.copy(); a2[cut:] *= rng.uniform(0.5, 1.5, len(a) - cut)
    z1 = S.ma_signal(S.ratio_series(a, b)[:, None], 35, 140)
    z2 = S.ma_signal(S.ratio_series(a2, b)[:, None], 35, 140)
    assert np.allclose(np.nan_to_num(z1[:cut], nan=-9), np.nan_to_num(z2[:cut], nan=-9))
    s1 = E.states(z1, 1.0, 0.25); s2 = E.states(z2, 1.0, 0.25)
    assert np.array_equal(s1[:cut], s2[:cut])


def test_crossing_rate_zero_and_nan_rules():
    x = np.array([1, 0, -1, np.nan, -2, 0, 0, -1, 3.0])[:, None]
    # transitions among non-missing obs: 8 values -> 7; crossings: 1->-1 (via 0) and -1->3
    assert S.crossing_rate(x)[0] == pytest.approx(2 / 7)
    assert S.crossing_rate(np.array([[1.0], [0.0], [1.0]]))[0] == 0.0
    assert S.crossing_rate(np.ones((50, 1)))[0] == 0.0
    assert S.crossing_rate(np.tile([1.0, -1.0], 50)[:, None])[0] > 0.98


def test_engle_granger_matches_statsmodels():
    a, b = _pair(400)
    p, beta, _ = S.engle_granger(np.log(a)[:, None], np.log(b)[:, None])
    ref = coint(np.log(a), np.log(b), trend="c", maxlag=1, autolag=None)[1]
    assert p[0] == pytest.approx(ref, abs=2e-3)


def test_engine_rules():
    z = np.array([0, -1.2, -0.8, -0.2, 0.0, 1.3, 2.0, -1.5, 0.3, 0.1])[:, None]
    s = E.states(z, 1.0, 0.25)[:, 0]
    #            open long, hold, exit, flat, short, no pyramid, reverse, exit (passes band), flat
    assert list(s) == [0, 1, 1, 0, 0, -1, -1, 1, 0, 0]
    # jump straight through the exit band closes the trade (original engine would hold)
    z2 = np.array([-1.2, 0.6, 0.6])[:, None]
    assert list(E.states(z2, 1.0, 0.25)[:, 0]) == [1, 0, 0]
    assert list(E.states(z2, 1.0, 0.25, "original")[:, 0]) == [1, 1, 1]
    # missing z holds the state and never opens
    z3 = np.array([np.nan, -1.5, np.nan, 0.0])[:, None]
    assert list(E.states(z3, 1.0, 0.25)[:, 0]) == [0, 1, 1, 0]


def test_direction_profits_on_reversion():
    # A cheap vs B -> long the ratio (buy A / sell B) earns when A recovers
    pa = np.r_[np.full(200, 100.0), np.full(8, 90.0), np.full(40, 100.0)]
    pb = np.full(len(pa), 100.0)
    r = pa / pb
    z = ((pd.Series(r) - pd.Series(r).rolling(30).mean()) / pd.Series(r).rolling(30).std()).to_numpy()[:, None]
    st = E.states(z, 1.0, 0.25)
    first = st[np.flatnonzero(st[:, 0] != 0)[0], 0]
    assert first == 1
    ret_a = np.r_[0, pa[1:] / pa[:-1] - 1]
    g = (np.r_[0, st[:-1, 0]] * ret_a / 2).sum()
    assert g > 0


def test_percentile_rank_orientation_and_ties():
    x = pd.Series([0.1, 0.5, 0.5, np.nan, 0.9])
    hi = SEL.pct_rank(x, True).to_numpy(); lo = SEL.pct_rank(x, False).to_numpy()
    assert hi[4] == 1.0 and hi[1] == hi[2] and hi[3] == 0.0
    assert lo[0] == 1.0 and lo[3] == 0.0


def test_selection_no_reuse():
    ia = np.array([0, 0, 1, 2, 3]); ib = np.array([1, 2, 2, 3, 4])
    sc = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    ch = SEL.select(sc, ia, ib, 3)
    names = np.r_[ia[ch], ib[ch]]
    assert len(names) == len(set(names)) and list(ch) == [0, 3]


def test_weights_never_exceed_capital_and_cap():
    pos = rng.choice([-1.0, 0.0, 1.0], size=(200, 20))
    phi = rng.uniform(0.5, 1.0, 20)
    for rule in ("active", "capped", "rank", "equal"):
        w = E.pair_weights(pos, phi, rule, 0.1)
        assert (w.sum(1) <= 1 + 1e-12).all() and (w >= 0).all()
        assert (w[pos == 0] == 0).all()
    assert E.pair_weights(pos, phi, "capped", 0.1).max() <= 0.1 + 1e-12


def test_book_round_trip_cost_and_neutrality():
    K = 6
    pos = np.array([1, 1, 1, 0, 0, 0.0])[:, None]
    w = np.ones((K, 1))
    ra = np.zeros((K, 1)); rb = np.zeros((K, 1))
    res = E.run_book(pos, w, ra, rb, np.array([0.5]), np.array([1e-4]), np.array([1e-4]), 0.0)
    # open 0.5 + 0.5, close 0.5 + 0.5 -> 2 dollars traded at 1 bp
    assert res["turn"].sum() == pytest.approx(2.0)
    assert res["cost"].sum() == pytest.approx(2e-4)
    # flat at end -> nothing left to liquidate beyond the close at bar 3
    assert res["turn"][-1] == 0.0


def test_forced_liquidation_and_drift():
    K = 4
    pos = np.ones((K, 1)); w = np.ones((K, 1))
    ra = np.full((K, 1), 0.01); rb = np.zeros((K, 1))
    res = E.run_book(pos, w, ra, rb, np.array([0.5]), np.zeros(1), np.zeros(1), 0.0)
    # no rebalancing while the target is unchanged: turnover only at open and final close
    assert res["turn"][1:-1].sum() == 0.0 and res["turn"][-1] > 0.99
    assert res["gross"].sum() > 0


def test_borrow_charged_on_short_leg():
    pos = np.ones((10, 1)); w = np.ones((10, 1)); z = np.zeros((10, 1))
    res = E.run_book(pos, w, z, z, np.array([0.5]), np.zeros(1), np.zeros(1), 1e-4)
    assert res["borrow"].sum() == pytest.approx(10 * 0.5 * 1e-4)


def test_protocol_grid_is_one_at_a_time():
    base = C.PRIMARY
    for c in C.grid():
        diffs = [f for f in base.__dataclass_fields__ if f != "label"
                 and getattr(c, f) != getattr(base, f)]
        assert 1 <= len(diffs) <= 3, (c.label, diffs)
