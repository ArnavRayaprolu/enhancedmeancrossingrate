"""CNN trading policy for pairs (after Guijarro-Ordonez, Pelger and Zanotti, "Deep
Learning Statistical Arbitrage", and Eggebrecht and Luetkebohmert 2023).

Instead of fixed z-score thresholds, a small 1-D convolutional network reads the
last L days of a pair's spread and outputs a position w in [-1, 1]
(+1 = long the ratio: buy A, sell B). It is trained, inside each walk-forward
fold and on formation data only, to maximise the Sharpe ratio of an
equal-weighted portfolio of training pairs net of a turnover cost.

Inputs per pair and day t (2 channels x L days, all known at the decision time):
    ch0  static z: (log ratio - formation mean) / formation sd
    ch1  rolling z: (log ratio - 20-day mean) / 20-day sd
Network: Conv1d(2->8, k=5) - ReLU - Conv1d(8->8, k=5) - ReLU -
         [mean pool, last value] - Linear(16->16) - ReLU - Linear(16->1) - tanh
Training: Adam, weight decay, full-batch over days, early stopping on the last
20% of formation days (validation Sharpe), at most 300 epochs.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(2)
L_WIN = 30
ROLL = 20


class Net(nn.Module):
    def __init__(self, ch=2, k=8, h=16):
        super().__init__()
        self.c1 = nn.Conv1d(ch, k, 5); self.c2 = nn.Conv1d(k, k, 5)
        self.f1 = nn.Linear(2 * k, h); self.f2 = nn.Linear(h, 1)

    def forward(self, x):                      # x: (B, ch, L)
        y = torch.relu(self.c1(x)); y = torch.relu(self.c2(y))
        y = torch.cat([y.mean(-1), y[..., -1]], 1)
        return torch.tanh(self.f2(torch.relu(self.f1(y)))).squeeze(-1)


def features(lr: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """lr: (T, n) log ratio up to and including each decision day. Returns
    (T, n, 2) feature values per day (NaN where the rolling window is short)."""
    z0 = (lr - mu) / sd
    T = lr.shape[0]
    z1 = np.full_like(lr, np.nan)
    for t in range(ROLL - 1, T):
        w = lr[t - ROLL + 1:t + 1]
        z1[t] = (lr[t] - w.mean(0)) / np.maximum(w.std(0), 1e-8)
    return np.stack([z0, z1], -1)


def windows(feat: np.ndarray, t_idx: np.ndarray) -> np.ndarray:
    """feat (T, n, 2) -> (len(t_idx), n, 2, L) windows ending at each t."""
    out = np.stack([feat[t - L_WIN + 1:t + 1] for t in t_idx], 0)   # (m, L, n, 2)
    return np.nan_to_num(np.transpose(out, (0, 2, 3, 1)), nan=0.0).clip(-6, 6)


def train(lr_f: np.ndarray, cost_rt: np.ndarray, seed: int = 0, epochs: int = 300):
    """lr_f: (F, n) formation log ratios of the training pairs; cost_rt: (n,)
    round-trip cost per unit gross. Returns a trained Net (eval mode) and the
    formation mean / sd used for the static channel."""
    torch.manual_seed(seed); np.random.seed(seed)
    mu, sd = lr_f.mean(0), np.maximum(lr_f.std(0), 1e-8)
    feat = features(lr_f, mu, sd)
    F = lr_f.shape[0]
    t_idx = np.arange(max(L_WIN, ROLL) - 1, F - 1)              # decision days with a next day
    X = torch.tensor(windows(feat, t_idx), dtype=torch.float32)  # (m, n, 2, L)
    R = torch.tensor((lr_f[t_idx + 1] - lr_f[t_idx]) / 2, dtype=torch.float32)  # (m, n)
    C = torch.tensor(cost_rt / 2, dtype=torch.float32)          # cost per unit |dw|
    m, n = R.shape
    split = int(m * 0.8)
    net = Net()
    opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-4)

    def sharpe(lo, hi):
        w = net(X[lo:hi].reshape(-1, 2, L_WIN)).reshape(hi - lo, n)
        dw = torch.cat([w[:1], w[1:] - w[:-1]], 0).abs()
        p = (w * R[lo:hi] - dw * C).mean(1)                    # equal-weighted portfolio, per day
        return p.mean() / (p.std() + 1e-6)

    best, best_state, bad = -1e9, None, 0
    for ep in range(epochs):
        net.train(); opt.zero_grad()
        loss = -sharpe(0, split); loss.backward(); opt.step()
        if ep % 5 == 0:
            net.eval()
            with torch.no_grad():
                v = float(sharpe(split, m))
            if v > best + 1e-4:
                best, best_state, bad = v, {k: x.clone() for k, x in net.state_dict().items()}, 0
            else:
                bad += 1
                if bad >= 8:
                    break
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    return net


def positions(net, lr_close_all: np.ndarray, lr_sig: np.ndarray, d0: int, mu, sd) -> np.ndarray:
    """Positions for the test window, decided causally.
    lr_close_all: (D, n) log ratio of daily closes (formation + test days);
    lr_sig: (K, n) log ratio at the signal price of decision days d0, d0+1, ...;
    the decision on day d = d0 + k uses closes up to day d-1 plus the signal price
    of day d (the close of day d is not yet known)."""
    K, n = lr_sig.shape
    M = max(L_WIN, ROLL) + 1
    out = np.zeros((K, n))
    with torch.no_grad():
        for k in range(K):
            d = d0 + k
            seq = np.vstack([lr_close_all[d - M:d], lr_sig[k:k + 1]])
            feat = features(seq, mu, sd)
            x = windows(feat, np.array([seq.shape[0] - 1]))[0]          # (n, 2, L)
            out[k] = net(torch.tensor(x, dtype=torch.float32)).numpy()
    return out
