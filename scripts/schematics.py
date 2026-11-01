"""Schematic figures for the paper: pipeline flowchart and leg assignment."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from emcr.figures import BLUE, INK, INK2, ORANGE  # noqa: E402

FIG = Path(__file__).resolve().parents[1] / "paper" / "figures"
FILL = {"data": "#eef4fc", "mcr": "#fde8df", "bench": "#f3f2ef", "engine": "#fff6dd"}


def box(ax, x, y, w, h, text, kind="data", fs=8.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                                fc=FILL[kind], ec="#52514e", lw=0.9))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK,
            fontweight="bold" if bold else "normal", linespacing=1.35)


def arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=9,
                                 color="#52514e", lw=0.9, shrinkA=0, shrinkB=0))


def flowchart():
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.5, 0.975, "Formation window (630 bars) — selection only", ha="center",
            fontsize=9, style="italic", color=INK2)
    box(ax, 0.02, 0.80, 0.29, 0.13, "Crypto venues, bars ≤ 1h\nBinance, Coinbase,\n"
        "Bitstamp, OKX", "data")
    box(ax, 0.355, 0.80, 0.29, 0.13, "Point-in-time universe\ntop-40 by formation\n"
        "dollar volume, ≥95% coverage", "data")
    box(ax, 0.69, 0.80, 0.29, 0.13, "Candidate pairs\nall pairs of eligible coins\n"
        "(identical for every criterion)", "data")
    arrow(ax, 0.31, 0.865, 0.355, 0.865); arrow(ax, 0.645, 0.865, 0.69, 0.865)
    box(ax, 0.02, 0.56, 0.45, 0.17, "Price ratio $r_t=P^A_t/P^B_t$\n"
        "MCR: share of bars at which $r_t$ crosses\nits 140-bar trailing mean\n"
        "CCR: completed 2-SD-to-mean cycles per bar", "mcr")
    box(ax, 0.53, 0.56, 0.45, 0.17, "Benchmarks\nDistance (SSD), Correlation,\n"
        "Engle–Granger cointegration, Half-life\nHybrids: ½ rk(MCR or CCR) + "
        "½ rk(Coint.)", "bench")
    ax.plot([0.835, 0.835, 0.5], [0.80, 0.765, 0.765], color="#52514e", lw=0.9)
    arrow(ax, 0.245, 0.765, 0.245, 0.73); arrow(ax, 0.755, 0.765, 0.755, 0.73)
    ax.plot([0.245, 0.755], [0.765, 0.765], color="#52514e", lw=0.9)
    box(ax, 0.20, 0.40, 0.60, 0.10, "Rank pairs by each criterion; select the top 5,\n"
        "no coin used twice", "data", bold=False)
    arrow(ax, 0.245, 0.56, 0.40, 0.50); arrow(ax, 0.755, 0.56, 0.60, 0.50)
    ax.plot([0.02, 0.98], [0.355, 0.355], color=INK2, lw=0.8, ls=(0, (4, 3)))
    ax.text(0.5, 0.31, "Trading window (315 bars) — strictly out of sample", ha="center",
            fontsize=9, style="italic", color=INK2)
    box(ax, 0.02, 0.06, 0.45, 0.20, "Trading rule (identical for all criteria)\n"
        "$z_t=(r_t-\\mathrm{MA}_{140})/\\mathrm{SD}_{140}$\n"
        "enter $|z_t|\\geq 2$, exit when $r_t$ is back at its mean\n"
        "start flat, liquidate at window end", "engine")
    box(ax, 0.53, 0.06, 0.45, 0.20, "Capital and costs\nrank-weighted across open pairs, "
        "unlevered\nfee + half-spread per \\$ traded, per leg\n"
        "drift rebalancing and short financing charged", "engine")
    arrow(ax, 0.40, 0.40, 0.245, 0.26); arrow(ax, 0.60, 0.40, 0.755, 0.26)
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "fig_methodology.pdf", bbox_inches="tight"); plt.close(fig)


def legs():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.2))
    for ax, (cond, buy, sell, title) in zip(axes, (
            ("$z_t\\leq-2$  (ratio far below its mean)", "BUY coin A", "SELL coin B", "long ratio"),
            ("$z_t\\geq+2$  (ratio far above its mean)", "BUY coin B", "SELL coin A", "short ratio"))):
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
        ax.text(0.5, 0.95, cond, ha="center", fontsize=9, color=INK)
        ax.add_patch(FancyBboxPatch((0.04, 0.52), 0.42, 0.26, boxstyle="round,pad=0.01",
                                    fc="#e3f1e6", ec="#52514e", lw=0.9))
        ax.text(0.25, 0.65, f"{buy}\n$+w/2$ of capital", ha="center", va="center", fontsize=8)
        ax.add_patch(FancyBboxPatch((0.54, 0.52), 0.42, 0.26, boxstyle="round,pad=0.01",
                                    fc="#f8e1df", ec="#52514e", lw=0.9))
        ax.text(0.75, 0.65, f"{sell}\n$-w/2$ of capital", ha="center", va="center", fontsize=8)
        ax.add_patch(FancyBboxPatch((0.14, 0.04), 0.72, 0.26, boxstyle="round,pad=0.01",
                                    fc="#f3f2ef", ec="#52514e", lw=0.9))
        ax.text(0.5, 0.17, "net dollar exposure = 0\nexit when the ratio is back at its mean",
                ha="center", va="center", fontsize=8)
        arrow(ax, 0.25, 0.52, 0.42, 0.30); arrow(ax, 0.75, 0.52, 0.58, 0.30)
    fig.savefig(FIG / "fig_legs.pdf", bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    flowchart(); legs(); print("ok")
