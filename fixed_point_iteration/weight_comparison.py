#!/usr/bin/env python
"""Hampel's fixed point against Tukey's, on identical samples, for all eight laws.

All starts converge to the same root, so one start (the sample median) suffices. The two
weights are evaluated on the very same samples, so the comparison is paired and the ratio
of (1-delta)-quantiles can be bootstrapped by resampling trials.
"""
import json, sys, time
from functools import partial

import numpy as np
from joblib import Parallel, delayed

sys.path.insert(0, "..")
from Experiment import shrinkage_functions_dict                       # noqa: E402
from fixed_point_experiment import (DELTA, ETA, N, K, DISTS, WEIGHTS,  # noqa: E402
                                    alpha_hat, draw, moments, BracketFailure)

T, B = 250_000, 4_000


def chunk(X, p):
    wfs = [partial(shrinkage_functions_dict[w], p=p) for w in WEIGHTS]
    out = np.full((len(X), len(WEIGHTS)), np.nan)
    for t, x in enumerate(X):
        for wi, wf in enumerate(wfs):
            kap = np.median(x)
            try:
                for _ in range(K):
                    d = np.abs(x - kap)
                    kap = kap + np.mean((x - kap) * wf(alpha_hat(d, wf, N, ETA) * d))
            except (BracketFailure, ValueError):
                kap = np.nan
            out[t, wi] = kap
    return out


def main(n_jobs=10):
    t0, rows = time.time(), {}
    rng = np.random.default_rng(11)
    print(f"{'law':5s}{'Hampel q':>12s}{'Tukey q':>12s}{'ratio H/T':>11s}"
          f"{'95% CI of ratio':>22s}{'winner':>9s}")
    for lab, a, r, d in DISTS:
        mu, nu2, p = moments(a, r, d)
        X = draw(a, r, d, T)
        res = Parallel(n_jobs=n_jobs)(
            delayed(chunk)(X[ii], p) for ii in np.array_split(np.arange(T), n_jobs * 3))
        kap = np.concatenate(res, axis=0)
        e = np.abs(kap - mu) / nu2                       # (T, 2), paired
        q = np.quantile(e, 1 - DELTA, axis=0, method="nearest")
        # Resample in batches: the (B, T) index matrix is 8 GB at T = 2.5*10^5, which is
        # what killed the delta = 2e-4 run. Cap each batch's temporaries at ~200 MB.
        per = max(1, int(2e8 // (T * e.shape[1] * 8)))
        ratio = np.empty(B)
        for i0 in range(0, B, per):
            m = min(per, B - i0)
            idx = rng.integers(0, T, size=(m, T))
            bq = np.quantile(e[idx], 1 - DELTA, axis=1, method="nearest")  # (m, 2)
            ratio[i0:i0 + m] = bq[:, 0] / bq[:, 1]
            del idx, bq
        lo, hi = np.quantile(ratio, [0.025, 0.975])
        win = "Hampel" if hi < 1 else ("Tukey" if lo > 1 else "tie")
        rows[lab] = {"hampel_q": float(q[0]), "tukey_q": float(q[1]),
                     "ratio": float(q[0] / q[1]), "ci": [float(lo), float(hi)], "winner": win}
        print(f"{lab:5s}{q[0]:12.5f}{q[1]:12.5f}{q[0] / q[1]:11.4f}"
              f"{f'[{lo:.4f}, {hi:.4f}]':>22s}{win:>9s}")
        del X, kap, e
    json.dump({"T": T, "n_bootstrap": B, "start": "sample median", "K": K, "by_law": rows,
               "runtime_sec": round(time.time() - t0, 1)},
              open("weight_comparison.json", "w"), indent=2)
    print(f"\n  runtime {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
