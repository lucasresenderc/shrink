from itertools import product
import json
from tqdm import tqdm
import numpy as np
from joblib import Parallel, delayed
from distributions import SkeGTD
from functools import partial
import polars as pl

CONTAMINATION = 1e6


def get_tm(X, k):
    assert 2*k < X.size
    return np.mean(np.sort(X)[k:-k])


def get_mom(X, K):
    block_len = X.size // K
    assert block_len > 0
    return np.median([np.mean(X[i*block_len:(i+1)*block_len]) for i in range(K)])


base_estimators_dict = {
    "mean": lambda X, delta, eps: np.mean(X),
    "median": lambda X, delta, eps: np.median(X),
    "mom": lambda X, delta, eps: get_mom(X, int(np.ceil(np.log(1/delta))+2*eps*len(X))),
    "tm": lambda X, delta, eps: get_tm(X, int(np.ceil(np.log(1/delta))+eps*len(X))),
}

shrinkage_functions_dict = {
    "lv": lambda t, p: (1 - t**p)*(t <= 1),
    "atm": lambda t, p: 1*(t <= 1),
    "win": lambda t, p: 1*(t <= 1) + np.nan_to_num((t > 1)/t),
    "exp": lambda t, p: np.exp(-t**p),
    "inv": lambda t, p: 1/(t**p + 1),
    # shrinkage functions that violate assumptions
    "ln_sq": lambda t, p: 1/np.log(np.e + t**2),
    "ln": lambda t, p: 1/np.log(np.e + t),
    "sqrt": lambda t, p: 1 - np.sqrt(1 - np.power((1 - t)*(t < 1), 2)),
    "inv_sqrt": lambda t, p: 1/(1 + np.sqrt(t)),
}

tolerance_dict = {
    "lv": 0.1,
    "atm": 1.1,
    "win": 0.1,
    "exp": 0.1,
    "inv": 0.1,
    "ln": 0.1,
    "ln_sq": 0.1,
    "sqrt": 0.1,
    "inv_sqrt": 0.1,
}


def fetch_moment(sample_dist):
    a, r, n, contamination_level = sample_dist
    return 2*a-0.0001 if a <= 1 else 2


def est_func(base_est, w, sym, norm, delta, c_eta, split, x, p, contamination_level, tol=1e-6):
    if split is None or w is None:
        kappa = base_estimators_dict[base_est](
            x, delta, contamination_level)
    else:
        if isinstance(split, float):
            split_idx = int(len(x)*split)
        elif isinstance(split, int):
            split_idx = split
        else:
            raise ValueError(
                "Split must be either float or int.")
        base_est_x = x[:split_idx]
        kappa = base_estimators_dict[base_est](
            base_est_x, delta, contamination_level)
    assert kappa is not None
    assert np.isfinite(kappa)
    if w is None:
        return kappa
    else:
        if split is None:
            shrink_x = x
        else:
            shrink_x = x[split_idx:]
        w_func = partial(shrinkage_functions_dict[w], p=p)
        # solve alpha such that
        # n - eta - 1 < sum rho(alpha |x - kappa|^p) <= n - eta
        n = shrink_x.size
        if not sym:
            eta = c_eta * np.log(1/delta) + contamination_level * n
            D = np.abs(shrink_x-kappa)
            if w != "atm":
                # we start finding an upper and a lower bound
                a = 0
                A = eta/n
                while np.sum(w_func(A * D)) > n - eta:
                    a, A = A, 2*A

                # now we fit
                new_a = (a+A)/2
                while A - a > tol or np.sum(w_func(new_a * D)) > n - eta:
                    if np.sum(w_func(new_a * D)) < n - eta:
                        A = new_a
                    else:
                        a = new_a
                    new_a = (a + A)/2
                if abs(n-eta-np.sum(w_func(new_a * D))) > tolerance_dict[w]:
                    raise ValueError(
                        f"Did not converge properly for w={w}, sym={sym}, norm={norm}, delta={delta}, c_eta={c_eta}, split={split}, contamination_level={contamination_level}, shrink_n={n}, eta={eta}, sum_weights={np.sum(w_func(new_a * D))}, a={a}, A={A}, new_a={new_a}, sum_a={np.sum(w_func(a * D))}, sum_A={np.sum(w_func(A * D))}")
                W = w_func(new_a * D)

            else:
                idx = np.argsort(D)[:int(np.floor(n - eta))]
                W = np.zeros_like(D)
                W[idx] = 1.0
            if norm:
                return np.sum(shrink_x * W)/np.sum(W)
            else:
                return kappa + np.mean((shrink_x-kappa) * W)
        else:
            eta = 2*c_eta * np.log(1/delta) + 2 * contamination_level * n
            D_p = (shrink_x - kappa)*(shrink_x >= kappa)
            D_n = (kappa - shrink_x)*(shrink_x < kappa)
            assert (D_p >= 0).all()
            assert (D_n >= 0).all()
            assert (D_p * D_n == 0).all()
            assert (D_p + D_n == np.abs(shrink_x - kappa)).all()
            assert np.sum(D_p > 0) > eta/2
            assert np.sum(D_n > 0) > eta/2
            a_p = 0
            A_p = eta/n
            while np.sum(w_func(A_p * D_p)) > n - eta/2:
                a_p, A_p = A_p, 2*A_p
            new_a_p = (a_p + A_p)/2
            while A_p - a_p > tol or np.sum(w_func(new_a_p * D_p)) > n - eta/2:
                if np.sum(w_func(new_a_p * D_p)) < n - eta/2:
                    A_p = new_a_p
                else:
                    a_p = new_a_p
                new_a_p = (a_p + A_p)/2
            a_n = 0
            A_n = eta/n
            while np.sum(w_func(A_n * D_n)) > n - eta/2:
                a_n, A_n = A_n, 2*A_n
            new_a_n = (a_n + A_n)/2
            while A_n - a_n > tol or np.sum(w_func(new_a_n * D_n)) > n - eta/2:
                if np.sum(w_func(new_a_n * D_n)) < n - eta/2:
                    A_n = new_a_n
                else:
                    a_n = new_a_n
                new_a_n = (a_n + A_n)/2
            W = w_func(new_a_p * D_p) + w_func(new_a_n * D_n) - 1
            try:
                assert np.isfinite(W).all()
            except AssertionError:
                raise ValueError(
                    "W contains non-finite values.")
            assert (W >= 0).all()

            if norm:
                return np.sum(shrink_x * W)/np.sum(W)
            else:
                return kappa + np.mean((shrink_x-kappa) * W)


class Experiment:
    def generate_sample(self, sample_dist, rng):
        a, r, n, contamination_level = sample_dist
        dist = SkeGTD(a=a, r=r, rng=rng)

        X = dist.rvs((self.n_trials, n))
        indices = np.tile(np.arange(n)[np.newaxis, :], (self.n_trials, 1))
        rng.permuted(indices, axis=1)
        corrupted_indices = indices[:, :int(np.floor(contamination_level * n))]
        X[corrupted_indices] = CONTAMINATION

        return X, np.float64(dist.mean())

    def __init__(self, name, base_estimators, shrinkage_function, symmetrized, normalized, deltas, c_etas, splits, dist_a, dist_r, ns, contamination_level, n_trials, seed=0, n_jobs=1):
        self.name = name
        self.base_estimators = base_estimators
        self.shrinkage_function = shrinkage_function
        self.symmetrized = symmetrized
        self.normalized = normalized
        self.deltas = deltas
        self.c_etas = c_etas
        self.splits = splits
        self.est_prod = list(product(base_estimators, shrinkage_function,
                                     symmetrized, normalized, deltas, c_etas, splits))

        self.dist_a = dist_a
        self.dist_r = dist_r
        self.ns = ns
        self.contamination_level = contamination_level
        self.dist_prod = product(dist_a, dist_r, ns, contamination_level)

        self.n_trials = n_trials
        self.n_jobs = n_jobs
        self.df = None
        self.seed = seed

    @ staticmethod
    def load_from_json(self, filename):
        specs = json.load(open(filename, "r"))
        self.metrics_df = pl.read_parquet(specs["metrics_filename"])
        self.name = specs["name"]
        self.specs_cols = specs["specs_cols"]

    def run_trial(self, sample_dist, seed):
        rng = np.random.default_rng(seed)
        X, true_mean = self.generate_sample(sample_dist, rng)
        base_estimates = []
        ws = []
        syms = []
        norms = []
        deltas = []
        c_etas = []
        splits = []
        estimates = []
        a_s = []
        rs = []
        ns = []
        contamination_levels = []
        true_means = []
        a = sample_dist[0]
        r = sample_dist[1]
        n = sample_dist[2]
        contamination_level = sample_dist[3]

        p = fetch_moment(sample_dist)

        for (base_est, w, sym, norm, delta, c_eta, split) in self.est_prod:
            estimates += [est_func(base_est, w, sym, norm,
                                   delta, c_eta, split, v, p, contamination_level) for v in X]
            a_s += self.n_trials*[a]
            rs += self.n_trials*[r]
            ns += self.n_trials*[n]
            contamination_levels += self.n_trials*[contamination_level]
            base_estimates += self.n_trials*[base_est]
            ws += self.n_trials*[w]
            syms += self.n_trials*[sym]
            norms += self.n_trials*[norm]
            deltas += self.n_trials*[delta]
            c_etas += self.n_trials*[c_eta]
            splits += self.n_trials*[split]
            true_means += self.n_trials*[true_mean]

        return {
            "a": a_s,
            "r": rs,
            "n": ns,
            "contamination_level": contamination_levels,
            "base_estimator": base_estimates,
            "shrinkage_function": ws,
            "is_symmetrized": syms,
            "is_normalized": norms,
            "delta": deltas,
            "c_eta": c_etas,
            "split_ratio": splits,
            "estimates": estimates,
            "true_mean": true_means,
        }

    def run(self):
        res = Parallel(n_jobs=self.n_jobs)(
            delayed(self.run_trial)(sample_dist, i + self.seed) for i, sample_dist in tqdm(enumerate(self.dist_prod), total=len(self.dist_a)*len(self.dist_r)*len(self.ns)*len(self.contamination_level))
        )
        dfs = [pl.DataFrame(r) for r in res]
        self.df = pl.concat(dfs)
        return self.df

    def save_experiment(self, filename):
        if self.df is None:
            raise ValueError(
                "No experiment data to save. Please run the experiment first.")
        self.df.write_parquet(filename)

    def compute_metrics(self, save=True):
        if self.df is None:
            raise ValueError(
                "No experiment data to compute metrics. Please run the experiment first.")
        self.df = self.df.with_columns(
            (pl.col("estimates")-pl.col("true_mean")).abs().alias("absolute_error"))
        self.specs_cols = [col for col in self.df.columns if col not in [
            "estimates", "true_mean", "absolute_error"]]
        self.metrics_df = self.df.group_by(self.specs_cols).agg([
            (pl.col("absolute_error")**2).mean().alias("mse"),
            (pl.col("absolute_error")).mean().alias("mae"),
            (pl.col("absolute_error").quantile(
                1-pl.col("delta").first())).alias("quantile")
        ])
        df_filename = "exp=" + self.name + "_metrics.parquet"
        json_filename = "exp=" + self.name + ".json"
        self.metrics_df.write_parquet(df_filename)
        dict_to_save = {
            "name": self.name,
            "specs_cols": self.specs_cols,
            "metrics_filename": df_filename
        }
        json.dump(dict_to_save, open(json_filename, "w"), indent=4)
