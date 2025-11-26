from itertools import product
import json
from tqdm import tqdm
import numpy as np
from joblib import Parallel, delayed
from distributions import SkeGTD
from functools import partial
import polars as pl

CONTAMINATION = 1e9


def get_tm(X, k):
    return np.mean(np.sort(X)[k:-k])


def get_mom(X, K):
    block_len = X.size // K
    return np.median([np.mean(X[i*block_len:(i+1)*block_len]) for i in range(K)])


base_estimators_dict = {
    "mean": lambda X, delta: np.mean(X),
    "median": lambda X, delta: np.median(X),
    "mom": lambda X, delta: get_mom(X, int(np.ceil(np.log(1/delta)))),
    "tm": lambda X, delta: get_tm(X, int(np.ceil(np.log(1/delta)))),
}

shrinkage_functions_dict = {
    "lv": lambda t, p: (1 - t**p)*(t <= 1),
    "atm": lambda t, p: 1*(t <= 1),
    "win": lambda t, p: 1*(t <= 1) + (t > 1)/t,
    "exp": lambda t, p: np.exp(-t**p),
    "inv": lambda t, p: 1/(t**p + 1),
}


def fetch_moment(sample_dist):
    a, r, n, contamination_level = sample_dist
    return 2*a-0.0001 if a <= 1 else 2


def est_func(base_est, w, sym, norm, delta, c_eta, split, x, p, contamination_level, tol=1e-6):
    if split is None or w is None:
        kappa = base_estimators_dict[base_est](x, delta)
    else:
        kappa = base_estimators_dict[base_est](x[:int(len(x)*split)], delta)
    if w is None:
        return kappa
    else:
        if split is None:
            shrink_x = x
        else:
            shrink_x = x[int(len(x)*split):]
        w_func = partial(shrinkage_functions_dict[w], p=p)
        # solve alpha such that
        # n - eta - 1 < sum rho(alpha |x - kappa|^p) <= n - eta
        n = shrink_x.size
        if not sym:
            eta = c_eta * np.log(1/delta) + contamination_level * n
            D = np.abs(shrink_x-kappa)

            # we start finding an upper and a lower bound
            a = 0
            A = eta/n
            while np.sum(w_func(A * D)) > n - eta:
                a, A = A, 2*A

            # now we fit
            while A - a > tol:
                if np.sum(w_func((A+a)/2 * D)) < n - eta:
                    A = (A+a)/2
                else:
                    a = (A+a)/2

            # now get the weights
            if norm:
                return np.sum(shrink_x * w_func(a * D))/np.sum(w_func(a * D))
            else:
                return kappa + np.mean((shrink_x-kappa) * w_func(a * D))
        else:
            eta = 2*c_eta * np.log(1/delta) + 2 * contamination_level * n
            D_p = (shrink_x - kappa)*(shrink_x >= kappa)
            D_n = (kappa - shrink_x)*(shrink_x < kappa)
            a_p = 0
            A_p = eta/n
            while np.sum(w_func(A_p * D_p)) > n - eta/2:
                a_p, A_p = A_p, 2*A_p
            while A_p - a_p > tol:
                if np.sum(w_func((A_p+a_p)/2 * D_p)) < n - eta/2:
                    A_p = (A_p+a_p)/2
                else:
                    a_p = (A_p+a_p)/2
            a_n = 0
            A_n = eta/n
            while np.sum(w_func(A_n * D_n)) > n - eta/2:
                a_n, A_n = A_n, 2*A_n
            while A_n - a_n > tol:
                if np.sum(w_func((A_n+a_n)/2 * D_n)) < n - eta/2:
                    A_n = (A_n+a_n)/2
                else:
                    a_n = (A_n+a_n)/2
            W = w_func(a_p * D_p) + w_func(a_n * D_n) - 1

            if norm:
                return np.sum(shrink_x * W)/np.sum(W)
            else:
                return kappa + np.mean((shrink_x-kappa) * W)


class Experiment:
    def generate_sample(self, sample_dist):
        rng = np.random.default_rng(self.seed)
        a, r, n, contamination_level = sample_dist
        dist = SkeGTD(a=a, r=r, rng=rng)

        X = dist.rvs((self.n_trials, n))
        corrupted_indices = rng.choice(n, size=int(
            contamination_level*n), replace=False
        )
        X[:, corrupted_indices] = CONTAMINATION

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
        self.seed = seed
        self.n_jobs = n_jobs
        self.df = None

    @staticmethod
    def load_from_json(self, filename):
        specs = json.load(open(filename, "r"))
        self.metrics_df = pl.read_parquet(specs["metrics_filename"])
        self.name = specs["name"]
        self.specs_cols = specs["specs_cols"]

    def run_trial(self, sample_dist):
        X, true_mean = self.generate_sample(sample_dist)
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
            delayed(self.run_trial)(sample_dist) for sample_dist in tqdm(self.dist_prod, total=len(self.dist_a)*len(self.dist_r)*len(self.ns)*len(self.contamination_level))
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
