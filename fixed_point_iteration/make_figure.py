#!/usr/bin/env python
"""Figures for section 8.3: one unimodal skewed law and one bimodal heavy-tailed law."""
import csv, collections
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, LogLocator, NullFormatter, ScalarFormatter

INK, MUTED, GRID, SPINE, REF = "#0b0b0b", "#52514e", "#e6e5e0", "#c3c2b7", "#8f8e88"
COLS = [("SN", "skewed Gaussian, one mode"), ("T2", r"Student $t$ (2.01), two modes")]
# Okabe–Ito blue / vermillion: darker than the previous pair, so they stay
# separable from the grey baselines and from each other in print and deuteranopia
SERIES = [("hampel", "Hampel (3-part)", "#0072B2", "solid"),
          ("tukey", "Tukey's biweight", "#D55E00", "solid")]
# top figure: colour = weight, dash = starting point
STARTS = [("median", "solid"), ("far", (0, (4, 1.6)))]
# baselines share one grey; dash distinguishes the three estimators
REFS = [("mean", "sample mean", (0, (1, 1.6))),
        ("median", "sample median", (0, (5, 2))),
        ("tm", "trimmed mean", "solid")]
KMAX = 5
FIGSIZE = (6.0, 2.1)

mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "mathtext.fontset": "cm", "font.size": 7.5, "axes.labelsize": 7, "axes.titlesize": 8,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "axes.linewidth": 0.8,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8, "xtick.major.pad": 1.5,
    "ytick.major.pad": 1.5, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.labelpad": 2, "axes.titlepad": 7, "axes.labelcolor": MUTED,
    "axes.edgecolor": SPINE, "pdf.fonttype": 42,
})

A, K0, B, TM = collections.defaultdict(dict), {}, collections.defaultdict(dict), {}
for r in csv.DictReader(open("panel_a.csv")):
    k = int(r["k"])
    A[(r["dist"], r["weight"], r["start"])][k] = float(r["err_quantile"])
    if k == 0 and r["weight"] == "hampel" and r["start"] in ("mean", "median"):
        K0[(r["dist"], r["start"])] = float(r["err_quantile"])   # k=0 of these starts IS the estimator
for r in csv.DictReader(open("panel_b.csv")):
    B[(r["dist"], r["weight"])][int(r["k"])] = (float(r["spread_p50"]), float(r["spread_p99"]))
P = collections.defaultdict(dict)
for r in csv.DictReader(open("panel_c.csv")):
    P[(r["dist"], r["weight"])][int(r["k"])] = (float(r["conv_p50"]), float(r["conv_p99"]))
for r in csv.DictReader(open("reference.csv")):
    TM[r["dist"]] = float(r["tm_quantile"])
KS = list(range(KMAX + 1))


def place(ys, lo, hi, frac=0.085):
    """Push label heights apart in log space; the gap is a fixed fraction of the panel."""
    gap = frac * np.log10(hi / lo)
    out = np.log10(np.array(ys, dtype=float))
    order = np.argsort(out)
    for i in range(1, len(order)):
        a, b = order[i - 1], order[i]
        if out[b] - out[a] < gap:
            out[b] = out[a] + gap
    return 10.0 ** out


def style(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(SPINE)
        ax.spines[sp].set_linewidth(0.8)


def new_fig():
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE, dpi=200, sharex=True,
                             gridspec_kw=dict(wspace=0.28))
    return fig, axes


def finish(fig, path):
    handles = [Line2D([], [], color=c, ls=dd, lw=1.6, label=l) for _w, l, c, dd in SERIES]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=2,
               frameon=False, fontsize=7, handlelength=2.6, columnspacing=2.0,
               labelcolor=MUTED, borderpad=0)
    # axes stop at 0.84 so a white band sits between the legend and the titles
    fig.tight_layout(pad=0.2, w_pad=0.3, rect=(0, 0, 1, 0.84))
    fig.savefig(f"{path}.pdf", bbox_inches="tight", pad_inches=0)
    fig.savefig(f"{path}.png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)


# --- (1-delta)-quantile of |kappa_k - mu| / nu2 ---------------------------------------
fig, axes = new_fig()
for j, (d, title) in enumerate(COLS):
    ax = axes[j]
    style(ax)
    rvals = [K0[(d, "mean")], K0[(d, "median")], TM[d]]
    for y, (_e, _nm, ls) in zip(rvals, REFS):
        ax.axhline(y, color=REF, lw=1.0, ls=ls, zorder=2)
    for w, _l, col, _d in SERIES:
        for st, dash in STARTS:
            ax.plot(KS, [A[(d, w, st)][k] for k in KS], color=col, lw=1.6, ls=dash,
                    zorder=3, solid_capstyle="round")
    ax.set_yscale("log")
    ax.set_title(title, color=INK)
    ax.set_xlabel(r"iteration $k$")
    ax.xaxis.set_major_locator(FixedLocator(KS))
    lv = rvals + [A[(d, w, st)][k] for w, *_ in SERIES for st, _ in STARTS
                  for k in range(2, KMAX + 1)]
    ax.set_ylim(0.80 * min(lv), 1.15 * max(rvals))
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5), numticks=12))
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="y", labelleft=True)
    if j == 0:
        ax.set_ylabel(r"$(1-\delta)$-quantile of $|\kappa_k-\mu|/\nu_2$" "\n"
                      r"against the iteration index")
    lo_, hi_ = ax.get_ylim()
    low = int(np.argmin(rvals))          # its label goes *below* the line: the plateau sits
    rest = [i for i in range(len(REFS)) if i != low]   # just above it and would collide
    ax.annotate(REFS[low][1], (0.06, rvals[low]), color=MUTED, fontsize=5.5,
                ha="left", va="top", xytext=(0, -1.6), textcoords="offset points")
    for i, y in zip(rest, place([rvals[i] for i in rest], lo_, hi_)):
        ax.annotate(REFS[i][1], (0.06, y), color=MUTED, fontsize=5.5, ha="left",
                    va="bottom", xytext=(0, 1.6), textcoords="offset points")

finish(fig, "fig_fixed_point")

# --- within-sample distance / nu2 ------------------------------------------------------
fig, axes = new_fig()
for j, (d, title) in enumerate(COLS):
    ax = axes[j]
    style(ax)
    for w, _l, col, dash in SERIES:
        p50 = [B[(d, w)][k][0] for k in KS]; p99 = [B[(d, w)][k][1] for k in KS]
        ax.fill_between(KS, p50, p99, color=col, alpha=0.18, lw=0, zorder=2)
        ax.plot(KS, p50, color=col, lw=1.6, ls=dash, zorder=3)
        tint = tuple(0.30 + 0.70 * c for c in mpl.colors.to_rgb(col))
        ax.plot(KS, [P[(d, w)][k][0] for k in KS], color=tint, lw=1.4,
                ls=(0, (3.5, 1.8)), zorder=4)
    ax.set_yscale("log")
    ax.set_title(title, color=INK)
    ax.set_xlabel(r"iteration $k$")
    ax.xaxis.set_major_locator(FixedLocator(KS))
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=7))
    ax.tick_params(axis="y", labelleft=True)
    if j == 0:
        ax.set_ylabel(r"within-sample distance $/\,\nu_2$")

finish(fig, "fig_fixed_point_distance")
print("wrote fig_fixed_point.pdf / .png")
print("wrote fig_fixed_point_distance.pdf / .png")
