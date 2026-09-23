#!/usr/bin/env python
"""Exhaustive search for fixed points of kappa -> mu-hat(kappa), per trial.

Every fixed point lies in [X_(1), X_(n)]: mu-hat(kappa) = (1 - S/n) kappa + (S/n) * (a
weighted mean of the sample) with 0 < S <= n - eta, so mu-hat(kappa) - kappa is > 0 below
X_(1) and < 0 above X_(n). We therefore sweep kappa over every order statistic and every
midpoint between consecutive ones -- a grid that adapts itself to the sample, and hence to
the heavy tails -- bracket every sign change of f = mu-hat - id, and bisect each to
machine precision. Reports the number of distinct roots per trial.
"""
import csv, json, sys, time
from functools import partial

import numpy as np
from joblib import Parallel, delayed

sys.path.insert(0, "..")
from Experiment import shrinkage_functions_dict          # noqa: E402
from fixed_point_iteration.fixed_point_experiment import (ETA, N, DISTS, WEIGHTS, alpha_hat, draw,  # noqa: E402
                                    moments, BracketFailure)

T_GRID = 1000
SEP = 1e-6          # two roots counted as distinct if they differ by more than SEP * nu2


def f_at(x, kap, wf, n):
    d = np.abs(x - kap)
    return kap + np.mean((x - kap) * wf(alpha_hat(d, wf, n, ETA) * d)) - kap


def roots_one(x, wf, n, nu2):
    xs = np.unique(np.sort(x))
    g = np.unique(np.concatenate([xs, 0.5 * (xs[:-1] + xs[1:])]))
    f = np.array([f_at(x, k, wf, n) for k in g])
    cand = list(g[np.abs(f) == 0.0])
    for i in np.where(f[:-1] * f[1:] < 0)[0]:
        lo, hi, flo = g[i], g[i + 1], f[i]
        for _ in range(100):
            m = 0.5 * (lo + hi)
            fm = f_at(x, m, wf, n)
            if fm == 0.0:
                lo = hi = m
                break
            if (fm > 0) == (flo > 0):
                lo, flo = m, fm
            else:
                hi = m
            if hi - lo <= 1e-15 * max(1.0, abs(lo)):
                break
        cand.append(0.5 * (lo + hi))
    keep = []
    for r in sorted(cand):
        if not keep or r - keep[-1] > SEP * nu2:
            keep.append(r)
    return len(keep), (np.diff(g).max() / nu2), (np.median(np.diff(g)) / nu2)


def chunk(X, nu2, p):
    wfs = [partial(shrinkage_functions_dict[w], p=p) for w in WEIGHTS]
    out = np.zeros((len(X), len(WEIGHTS)), dtype=np.int32)
    gmax = np.zeros(len(X)); gmed = np.zeros(len(X))
    for t, x in enumerate(X):
        for wi, wf in enumerate(wfs):
            try:
                out[t, wi], gx, gm = roots_one(x, wf, X.shape[1], nu2)
            except (BracketFailure, ValueError):
                out[t, wi] = -1
                gx = gm = np.nan
        gmax[t], gmed[t] = gx, gm
    return out, gmax, gmed


def main(n_trials=T_GRID, n_jobs=10):
    t0 = time.time()
    rows, summ = [], {}
    for lab, a, r, d in DISTS:
        mu, nu2, p = moments(a, r, d)
        X = draw(a, r, d, n_trials)
        idx = np.array_split(np.arange(n_trials), n_jobs * 3)
        res = Parallel(n_jobs=n_jobs)(delayed(chunk)(X[ii], nu2, p) for ii in idx)
        cnt = np.concatenate([q[0] for q in res], axis=0)
        gmax = np.concatenate([q[1] for q in res]); gmed = np.concatenate([q[2] for q in res])
        for wi, w in enumerate(WEIGHTS):
            c = cnt[:, wi]
            hist = {int(v): int((c == v).sum()) for v in np.unique(c)}
            summ[f"{lab}/{w}"] = {"trials": int(len(c)), "root_count_hist": hist,
                                  "frac_multiple": float(np.mean(c > 1)),
                                  "max_roots": int(c.max()), "failed": int((c < 0).sum())}
            rows.append(f"{lab},{w},{len(c)},{c.min()},{c.max()},{np.mean(c > 1):.6g},"
                        f"{np.mean(c):.6g},{np.median(gmed):.4g},{np.median(gmax):.4g}")
        print(f"  {lab:4s} roots/trial: hampel {summ[lab + '/hampel']['root_count_hist']}, "
              f"tukey {summ[lab + '/tukey']['root_count_hist']}   "
              f"grid step (median/max, nu2 units) {np.median(gmed):.2e}/{np.median(gmax):.2e}"
              f"   [{time.time() - t0:.0f}s]")
        del X
    open("grid_search.csv", "w").write(
        "dist,weight,trials,min_roots,max_roots,frac_multiple,mean_roots,"
        "grid_step_median,grid_step_max\n" + "\n".join(rows) + "\n")
    summ["_meta"] = {"T_grid": n_trials, "separation": SEP,
                     "grid": "order statistics and their midpoints",
                     "runtime_sec": round(time.time() - t0, 1)}
    json.dump(summ, open("grid_search.json", "w"), indent=2)
    tot = sum(v["trials"] for k, v in summ.items() if k != "_meta")
    mult = sum(v["frac_multiple"] * v["trials"] for k, v in summ.items() if k != "_meta")
    print(f"\n  {tot} (trial, weight) sweeps; trials with more than one fixed point: {mult:.0f}")
    print(f"  runtime {summ['_meta']['runtime_sec']}s")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else T_GRID,
         int(sys.argv[2]) if len(sys.argv) > 2 else 10)
