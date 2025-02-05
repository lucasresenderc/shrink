import numpy as np
from itertools import product
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from distributions import SkeGTD

def get_tm(X, k):
    return np.mean(np.sort(X)[k:-k])

def get_mom(X, K):
    l = X.size // K
    return np.median([np.mean(X[i*l:(i+1)*l]) for i in range(K)])

def get_mean(X, kappa, rho, eta, p, eps=0.00001):
    # solve alpha such that
    # n - eta - 1 < sum rho(alpha |x - kappa|^p) <= n - eta
    n = X.size
    w = np.power(np.abs(X-kappa), p)

    # we start finding an upper and a lower bound
    a = 0
    A = eta/n
    while np.sum(rho(A * w)) > n - eta:
        a, A = A, 2*A

    # now we fit
    while A - a > eps:
        if np.sum(rho((A+a)/2 * w)) > n - eta:
            A = (A+a)/2
        else:
            a = (A+a)/2

    # now get the weights
    return kappa + np.mean((X-kappa) * rho(a * w))


# assymetric trimmed mean
rho_atm = lambda x : 1*(x < 1)

# winsorized mean
rho_win = lambda x : 1*(x < 1) + (x >= 1)/(x+.000001)

# lee and valiant
rho_lv = lambda x : (1 - x)*(x < 1)

# personalized choices
rho_exp = lambda x : np.exp(-x)

# works only if p >= 2
rho_half = lambda x : 1*(x<.25) + (x < 1) * (x > .25) * (1-x) * 4/3
rho_third = lambda x : 1*(x<1/9) + (x < 1) * (x > 1/9) * (1-x) * 9/8
rho_quarter = lambda x : 1*(x<1/16) + (x < 1) * (x > 1/16) * (1-x) * 16/15

def generate_sample(n_trials, n, a, r, seed=1):
    rng = np.random.default_rng(seed)
    dist = SkeGTD(a=a, r=r, rng=rng)
    return dist.rvs((n_trials, n))

def task(params):
    n, delta, ((tail, skew), distribution) = params
    n_trials = 10000 # change here to improve
    X = generate_sample(n_trials, n, tail, skew)
    estimates = []
    ns = []
    methods = []
    deltas = []
    distributions = []
    estimates += [np.mean(v) for v in X]
    estimates += [get_tm(v, int(np.ceil(np.log(1/delta)))) for v in X]
    estimates += [get_mom(v, int(np.ceil(np.log(1/delta)))) for v in X]
    estimates += [get_mean(v[n//2:], np.mean(v[:n//2]), rho_atm, np.log(1/delta), 1) for v in X]
    estimates += [get_mean(v[n//2:], np.mean(v[:n//2]), rho_win, np.log(1/delta), 1) for v in X]

    p = 2*tail - 0.01 if tail <= 1 else 2
    estimates += [get_mean(v[n//2:], np.mean(v[:n//2]), rho_lv, np.log(1/delta), p) for v in X]
    estimates += [get_mean(v[n//2:], np.mean(v[:n//2]), rho_exp, np.log(1/delta), p) for v in X]
    estimates += [get_mean(v[n//2:], np.mean(v[:n//2]), rho_half, np.log(1/delta), p) for v in X]
    estimates += [get_mean(v[n//2:], np.mean(v[:n//2]), rho_third, np.log(1/delta), p) for v in X]
    estimates += [get_mean(v[n//2:], np.mean(v[:n//2]), rho_quarter, np.log(1/delta), p) for v in X]

    methods += n_trials*['mean'] + n_trials*['tm'] + n_trials*['mom'] + n_trials*['atm'] + n_trials*['win'] + n_trials*['lv']
    methods += n_trials*['exp'] + n_trials*['half'] + n_trials*['third'] + n_trials*['quarter']
    ns += 10*n_trials*[n]
    deltas += 10*n_trials*[delta]
    distributions += 10*n_trials*[distribution]

    return ns, methods, estimates, deltas, distributions

# range_ns = [n for n in range(50,500,25)]
# range_deltas = np.exp(-np.linspace(1,6,24))
# range_ts = [(np.inf, 0), (np.inf, .9), (.605, 0), (.605, .9), (1.005, 0), (1.005, .9)]
# range_distributions = ["Gaussian", "Skewed Gaussian", "St df=1.21", "Skewed St df=1.21", "St df=2.01",  "Skewed St df=2.01"]

range_ns = [n for n in range(50,500,50)]
range_deltas = [.1, .01, .001]
range_ts = [(np.inf, 0), (1.005, 0)]
range_distributions = ["Gaussian", "St df=2.01"]

loop_n = len(range_ns)*len(range_deltas)*len(range_ts)
ret = Parallel(n_jobs = 8)(delayed(task)(params) for params in tqdm(product(range_ns, range_deltas, zip(range_ts, range_distributions)), total=loop_n))

df = [pd.DataFrame({'n': r[0], 'method': r[1], 'estimate': r[2], 'delta': r[3], 'distribution': r[4]}) for r in ret]
df = pd.concat(df)
df.to_parquet("results.parquet")