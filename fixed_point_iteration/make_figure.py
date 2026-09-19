#!/usr/bin/env python
"""Figure for section 8.3: one unimodal skewed law and one bimodal heavy-tailed law."""
import csv, json, collections
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, LogLocator, NullFormatter, ScalarFormatter

INK, MUTED, GRID, SPINE, REF = "#0b0b0b", "#52514e", "#e6e5e0", "#c3c2b7", "#8f8e88"
COLS = [("SN", "skewed Gaussian, one mode"), ("T2", r"Student $t$ (2.01), two modes")]
SERIES = [("hampel", "Hampel (3-part)", "#2a78d6", "solid"),
          ("tukey", "Tukey's biweight", "#eb6834", "solid")]
# top row: colour = weight, dash = starting point
STARTS = [("median", "solid"), ("far", (0, (4, 1.6)))]
# baselines: own hue each, min OKLab dE 15.2 among themselves and 15.5 against the two
# series colours, under normal vision and under deutan and protan simulation
REFS = [("mean", "sample mean", "#46b8a1", (0, (1, 1.6))),
        ("median", "sample median", "#8c4635", (0, (5, 2))),
        ("tm", "trimmed mean", "#43437d", "solid")]
KMAX = 5

mpl.rcParams.update({
    "font.family": "serif", "font.serif": ["STIX Two Text", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 7.5, "axes.labelsize": 7, "axes.titlesize": 8,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "axes.linewidth": 0.8,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.labelcolor": MUTED, "axes.edgecolor": SPINE, "pdf.fonttype": 42,
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
G = json.load(open("grid_search.json"))
KS = list(range(KMAX + 1))

fig, axes = plt.subplots(2, 2, figsize=(5.4, 4.5), dpi=200, sharex=True)
axes[1, 1].sharey(axes[1, 0])          # row 2 is dimensionless; row 1 is not


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


for j, (d, title) in enumerate(COLS):
    ax = axes[0, j]
    ax.set_axisbelow(True); ax.grid(axis="y", color=GRID, lw=0.6)
    rvals = [K0[(d, "mean")], K0[(d, "median")], TM[d]]
    for y, (_e, _nm, rc, ls) in zip(rvals, REFS):
        ax.axhline(y, color=rc, lw=1.0, ls=ls, zorder=2)
    for w, _l, col, _d in SERIES:
        for st, dash in STARTS:
            ax.plot(KS, [A[(d, w, st)][k] for k in KS], color=col, lw=1.6, ls=dash,
                    zorder=3, solid_capstyle="round")
    ax.set_yscale("log"); ax.set_title(title, color=INK, pad=4)
    lv = rvals + [A[(d, w, st)][k] for w, *_ in SERIES for st, _ in STARTS
                  for k in range(2, KMAX + 1)]
    ax.set_ylim(0.80 * min(lv), 1.15 * max(rvals))
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5), numticks=12))
    ax.yaxis.set_major_formatter(ScalarFormatter()); ax.yaxis.set_minor_formatter(NullFormatter())
    if j == 0:
        ax.annotate("solid: started at the sample median\n"
                    r"dashed: started at $\mu+10\nu_2$",
                    (0.97, 0.42), xycoords="axes fraction", color=MUTED, fontsize=5.5,
                    ha="right", va="bottom", linespacing=1.5)
    lo_, hi_ = ax.get_ylim()
    low = int(np.argmin(rvals))          # its label goes *below* the line: the plateau sits
    rest = [i for i in range(len(REFS)) if i != low]   # just above it and would collide
    ax.annotate(REFS[low][1], (0.06, rvals[low]), color=REFS[low][2], fontsize=5.5,
                ha="left", va="top", xytext=(0, -1.6), textcoords="offset points")
    for i, y in zip(rest, place([rvals[i] for i in rest], lo_, hi_)):
        ax.annotate(REFS[i][1], (0.06, y), color=REFS[i][2], fontsize=5.5, ha="left",
                    va="bottom", xytext=(0, 1.6), textcoords="offset points")
    if j == 0:
        ax.set_ylabel(r"$(1-\delta)$-quantile of $|\kappa_k-\mu|/\nu_2$" "\n"
                      r"against the iteration index")

for j, (d, title) in enumerate(COLS):
    ax = axes[1, j]
    ax.set_axisbelow(True); ax.grid(axis="y", color=GRID, lw=0.6)
    for w, _l, col, dash in SERIES:
        p50 = [B[(d, w)][k][0] for k in KS]; p99 = [B[(d, w)][k][1] for k in KS]
        ax.fill_between(KS, p50, p99, color=col, alpha=0.18, lw=0, zorder=2)
        ax.plot(KS, p50, color=col, lw=1.6, ls=dash, zorder=3)
        tint = tuple(0.45 + 0.55 * c for c in mpl.colors.to_rgb(col))
        ax.plot(KS, [P[(d, w)][k][0] for k in KS], color=tint, lw=1.4,
                ls=(0, (3.5, 1.8)), zorder=4)
    ax.set_yscale("log")
    ax.set_xlabel(r"iteration $k$"); ax.xaxis.set_major_locator(FixedLocator(KS))
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=7))
    gt = sum(G[f"{d}/{w}"]["trials"] for w, *_ in SERIES)
    ax.annotate(f"separate grid sweep: one fixed point in {gt}/{gt} samples", (0.03, 0.045),
                xycoords="axes fraction", color=MUTED, fontsize=5.5, ha="left", va="bottom")
    if j == 0:
        ax.set_ylabel(r"within-sample distance $/\,\nu_2$")
        ax.annotate("solid + band: spread of the six starting points,\n"
                    r"$\max_{s,s'}|\kappa_k^{(s)}-\kappa_k^{(s')}|$, median to 99th pct"
                    "\npale dashed: sample-median start, distance\nto its own fixed point",
                    (0.97, 0.955), xycoords="axes fraction", color=MUTED, fontsize=5.5,
                    ha="right", va="top", linespacing=1.5)

for ax in axes.ravel():
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(SPINE); ax.spines[sp].set_linewidth(0.8)

handles = [Line2D([], [], color=c, ls=dd, lw=1.6, label=l) for _w, l, c, dd in SERIES]
fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2,
           frameon=False, fontsize=7, handlelength=2.6, columnspacing=2.0,
           labelcolor=MUTED, borderpad=0)
fig.tight_layout(rect=(0, 0, 1, 0.955))
fig.savefig("fig_fixed_point.pdf", bbox_inches="tight")
fig.savefig("fig_fixed_point.png", bbox_inches="tight")
print("wrote fig_fixed_point.pdf / .png")
