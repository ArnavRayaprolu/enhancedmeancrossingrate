"""Publication figures (static PDF, print-safe).

Colour roles: proposed hybrids take categorical slots 1-2 (blue, orange);
cointegration and other benchmarks are neutral grey / dashed; polarity
(Sharpe ratio) uses a blue-grey-red diverging scale centred at zero.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

from . import config as C  # noqa: E402

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8d8c86", "#e4e3df"
DIVERGING = LinearSegmentedColormap.from_list(
    "bgr", ["#104281", "#3987e5", "#cde2fb", "#f0efec", "#f6c8c4", "#e34948", "#8f1d1c"])

STYLE = {
    "MCR": dict(color=BLUE, lw=1.8, ls="-"),
    "MCR+Coint": dict(color=ORANGE, lw=1.6, ls="-"),
    "Cointegration": dict(color=INK2, lw=1.4, ls="--"),
}
LABEL = {"Distance": "Distance", "Correlation": "Correlation", "Cointegration": "Cointegration",
         "HalfLife": "Half-life", "MCR": "MCR", "CCR": "CCR",
         "MCR+Coint": "MCR+Coint", "CCR+Coint": "CCR+Coint"}
DS_LABEL = {k: v.label for k, v in C.DATASETS.items()}

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "legend.fontsize": 8,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "figure.dpi": 150,
    "savefig.bbox": "tight", "pdf.fonttype": 42,
})


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def sharpe_heatmap(perf: pd.DataFrame, path: Path, title: str = ""):
    """perf: rows = dataset, columns = criterion, values = net Sharpe."""
    cols = [c for c in C.CRITERIA if c in perf.columns]
    M = perf[cols]
    v = np.nanmax(np.abs(M.to_numpy())) or 1.0
    fig, ax = plt.subplots(figsize=(7.2, 0.42 * len(M) + 1.2))
    im = ax.imshow(M.to_numpy(), cmap=DIVERGING, norm=TwoSlopeNorm(0, -v, v), aspect="auto")
    ax.set_xticks(range(len(cols)), [LABEL.get(c, c) for c in cols], rotation=30, ha="right")
    ax.set_yticks(range(len(M)), [DS_LABEL.get(i, i) for i in M.index])
    ax.grid(False)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            x = M.iat[i, j]
            if np.isfinite(x):
                ax.text(j, i, f"{x:+.2f}", ha="center", va="center", fontsize=7.5,
                        color="white" if abs(x) > 0.6 * v else INK)
    ax.axvline(3.5, color="white", lw=2)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("net Sharpe ratio", color=INK2); cb.outline.set_visible(False)
    if title:
        ax.set_title(title, loc="left", color=INK)
    _save(fig, path)


def equity_panels(series: dict, path: Path, ncols: int = 2):
    """series: {dataset: {criterion: net return Series}} -> small multiples."""
    names = list(series)
    nrows = int(np.ceil(len(names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.2, 2.1 * nrows), squeeze=False)
    for ax, ds in zip(axes.ravel(), names):
        for m in ("Cointegration", "MCR+Coint", "MCR"):
            if m not in series[ds]:
                continue
            eq = (1 + series[ds][m]).cumprod()
            ax.plot(eq.index, eq.to_numpy(), label=LABEL.get(m, m), **STYLE[m])
            ax.annotate(f"{eq.iloc[-1] - 1:+.0%}", (eq.index[-1], eq.iloc[-1]),
                        xytext=(3, 0), textcoords="offset points", fontsize=7,
                        color=INK2, va="center")
        ax.axhline(1.0, color=MUTED, lw=0.8)
        ax.set_title(DS_LABEL.get(ds, ds), loc="left", color=INK)
        ax.tick_params(axis="x", labelsize=7)
        for lab in ax.get_xticklabels():
            lab.set_rotation(0)
        ax.xaxis.set_major_locator(matplotlib.dates.AutoDateLocator(maxticks=4))
        ax.xaxis.set_major_formatter(matplotlib.dates.ConciseDateFormatter(
            ax.xaxis.get_major_locator()))
    for ax in axes.ravel()[len(names):]:
        ax.set_visible(False)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.supylabel("growth of 1 (net)", color=INK2, fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    _save(fig, path)


def grid_differences(diffs: dict, path: Path):
    """diffs: {dataset: {benchmark: array of Sharpe(MCR) - Sharpe(benchmark) over configs}}."""
    comps = (("Distance", BLUE), ("Correlation", ORANGE), ("Cointegration", "#1baf7a"))
    names = list(diffs)
    fig, ax = plt.subplots(figsize=(7.2, 0.62 * len(names) + 1.0))
    for i, ds in enumerate(names):
        for off, (m, col) in zip((-0.22, 0.0, 0.22), comps):
            d = np.asarray(diffs[ds].get(m, []), float)
            d = d[np.isfinite(d)]
            if not len(d):
                continue
            q1, med, q3 = np.percentile(d, [25, 50, 75]); lo, hi = np.percentile(d, [5, 95])
            y = i + off
            ax.plot([lo, hi], [y, y], color=col, lw=1.0)
            ax.plot([q1, q3], [y, y], color=col, lw=4, solid_capstyle="butt", alpha=0.9)
            ax.plot(med, y, "o", ms=5, color="white", mec=col, mew=1.5, zorder=3)
            ax.text(hi, y, f"  {(d > 0).mean():.0%}", va="center", fontsize=6.5, color=INK2)
    ax.axvline(0, color=INK2, lw=0.9)
    ax.set_yticks(range(len(names)), [DS_LABEL.get(n, n) for n in names])
    ax.invert_yaxis()
    ax.set_xlabel("Sharpe(MCR) − Sharpe(benchmark), same configuration "
                  "(median, IQR, 5–95%; label = share > 0)")
    from matplotlib.lines import Line2D
    ax.legend([Line2D([], [], color=c, lw=4) for _, c in comps],
              [f"vs {m}" for m, _ in comps], loc="upper center", frameon=False, ncol=3,
              bbox_to_anchor=(0.5, -0.12 - 0.02 * len(names)))
    ax.grid(axis="y", visible=False)
    _save(fig, path)


def ic_dots(ic: pd.DataFrame, path: Path, outcome: str = "oos_net", signal: str = "LEVEL"):
    """Mean fold-level IC +/- 2 s.e. for each feature, one row per dataset."""
    feats = ["mcr", "ccr", "halflife", "ar1", "coint_p", "distance", "corr"]
    flab = {"mcr": "MCR", "ccr": "CCR", "halflife": "half-life", "ar1": "AR(1) of z",
            "coint_p": "coint. p", "distance": "distance", "corr": "correlation"}
    d = ic[(ic.outcome == outcome) & (ic.signal == signal)]
    names = list(dict.fromkeys(d["dataset"]))
    fig, axes = plt.subplots(1, len(names), figsize=(1.6 * len(names) + 1.2, 2.8),
                             sharey=True, squeeze=False)
    for ax, ds in zip(axes[0], names):
        g = d[d.dataset == ds].set_index("feature").reindex(feats)
        se = g["std"] / np.sqrt(g["count"])
        y = np.arange(len(feats))
        ax.errorbar(g["mean"], y, xerr=2 * se, fmt="o", ms=5, color=BLUE, ecolor=BLUE,
                    elinewidth=1.2, capsize=0)
        ax.axvline(0, color=INK2, lw=0.8)
        ax.set_title(DS_LABEL.get(ds, ds), color=INK, fontsize=8)
        ax.grid(axis="y", visible=False)
    axes[0, 0].set_yticks(np.arange(len(feats)), [flab[f] for f in feats])
    axes[0, 0].invert_yaxis()
    fig.supxlabel("mean fold-level rank IC with out-of-sample net pair return (±2 s.e.)",
                  color=INK2, fontsize=8)
    fig.tight_layout()
    _save(fig, path)


def cost_curves(cost: pd.DataFrame, path: Path):
    """cost: rows (dataset, criterion, cost_bps, sharpe) for the primary config."""
    names = list(dict.fromkeys(cost["dataset"]))
    ncols = min(4, len(names)); nrows = int(np.ceil(len(names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.2, 1.9 * nrows + 0.4), squeeze=False)
    for ax, ds in zip(axes.ravel(), names):
        for m in ("Cointegration", "MCR+Coint", "MCR"):
            g = cost[(cost.dataset == ds) & (cost.criterion == m)]
            ax.plot(g["cost_bps"], g["sharpe"], marker="o", ms=3, label=LABEL.get(m, m), **STYLE[m])
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.axvline(C.DATASETS[ds].cost_bps, color=MUTED, lw=0.8, ls=":")
        ax.set_title(DS_LABEL.get(ds, ds), loc="left", color=INK, fontsize=8)
    for ax in axes.ravel()[len(names):]:
        ax.set_visible(False)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.supxlabel("one-way cost per leg (bps); dotted line = base case", color=INK2, fontsize=8,
                  y=0.04)
    fig.supylabel("net Sharpe ratio", color=INK2, fontsize=8)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    _save(fig, path)


def walkforward(folds: pd.DataFrame, path: Path, title: str):
    fig, ax = plt.subplots(figsize=(5.2, 0.16 * len(folds) + 1.0))
    for i, r in folds.iterrows():
        ax.barh(i, (r.test_start - r.train_start), left=r.train_start, color="#b7d3f6",
                height=0.7, edgecolor="white", lw=0.6)
        ax.barh(i, (r.test_end - r.test_start), left=r.test_start, color="#184f95",
                height=0.7, edgecolor="white", lw=0.6)
    ax.invert_yaxis(); ax.set_ylabel("fold"); ax.grid(axis="y", visible=False)
    ax.set_title(title, loc="left", color=INK)
    from matplotlib.patches import Patch
    ax.legend([Patch(color="#b7d3f6"), Patch(color="#184f95")],
              ["formation (630 bars)", "trading (315 bars)"], frameon=False,
              loc="lower center", bbox_to_anchor=(0.5, -0.45), ncol=2)
    _save(fig, path)


def signal_example(z: pd.Series, pos: pd.Series, entry: float, exit_: float, path: Path,
                   title: str):
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    ax.axhspan(entry, max(entry, np.nanmax(z) + 0.2), color="#f6c8c4", alpha=0.6, lw=0)
    ax.axhspan(min(-entry, np.nanmin(z) - 0.2), -entry, color="#f6c8c4", alpha=0.6, lw=0)
    ax.axhspan(-exit_, exit_, color="#cde2fb", alpha=0.8, lw=0)
    ax.plot(z.index, z.to_numpy(), color=INK, lw=1.1)
    ax.axhline(0, color=INK2, lw=0.7)
    chg = pos.diff().fillna(pos)
    ent = chg[(chg != 0) & (pos != 0)].index
    ext = chg[(chg != 0) & (pos == 0)].index
    # markers at the decision bar (the bar before the position changes)
    zi = z.index
    dec = lambda ix: [zi[max(zi.get_loc(t) - 1, 0)] for t in ix]
    ax.plot(dec(ent), z.reindex(dec(ent)).to_numpy(), "^", ms=7, color=BLUE, mec="white",
            label="entry decision")
    ax.plot(dec(ext), z.reindex(dec(ext)).to_numpy(), "s", ms=6, color=ORANGE, mec="white",
            label="exit decision")
    ax.set_ylabel("z"); ax.set_title(title, loc="left", color=INK)
    ax.text(z.index[0], entry, "  entry region |z| ≥ %.2f" % entry, va="bottom",
            fontsize=7, color=INK2)
    ax.text(z.index[0], exit_, "  exit region |z| ≤ %.2f" % exit_, va="bottom",
            fontsize=7, color=INK2)
    ax.legend(frameon=False, loc="upper right", ncol=2)
    ax.xaxis.set_major_formatter(matplotlib.dates.ConciseDateFormatter(
        matplotlib.dates.AutoDateLocator()))
    _save(fig, path)
