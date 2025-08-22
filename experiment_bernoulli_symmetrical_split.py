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


def get_mean(X, kappa, rho, eta, p, eps=0.00001, sym=False):
    # solve alpha such that
    # n - eta - 1 < sum rho(alpha |x - kappa|^p) <= n - eta
    n = X.size
    w = np.power(np.abs(X-kappa), p)
    if sym:
        signs = np.random.choice([-1, 1], size=n)
    else:
        signs = np.ones(n)
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
    return kappa + np.mean(signs*(X-kappa) * rho(a * w))


# assymetric trimmed mean
def rho_atm(x): return 1*(x < 1)

# winsorized mean


def rho_win(x): return 1*(x < 1) + (x >= 1)/(x+.000001)

# lee and valiant


def rho_lv(x): return (1 - x)*(x < 1)

# personalized choices


def rho_exp(x): return np.exp(-x)


def generate_sample(n_trials, n, a, r, seed=1):
    rng = np.random.default_rng(seed)
    dist = SkeGTD(a=a, r=r, rng=rng)
    return dist.rvs((n_trials, n))


def task(params):
    n, delta, c_eta, ((tail, skew), distribution) = params
    n_trials = 10000  # change here to improve
    X = generate_sample(n_trials, n, tail, skew)
    estimates = []
    ns = []
    methods = []
    deltas = []
    distributions = []
    c_etas = []
    estimates += [np.mean(v) for v in X]
    estimates += [get_tm(v, int(np.ceil(c_eta*np.log(1/delta)))) for v in X]
    estimates += [get_mom(v, int(np.ceil(c_eta*np.log(1/delta)))) for v in X]
    estimates += [get_mean(v[n//2:], np.mean(v[:n//2]),
                           rho_atm, c_eta*np.log(1/delta), 1) for v in X]
    estimates += [get_mean(v[n//2:], np.mean(v[:n//2]),
                           rho_win, c_eta*np.log(1/delta), 1) for v in X]

    p = 2*tail - 0.01 if tail <= 1 else 2
    estimates += [get_mean(v[n//2:], np.mean(v[:n//2]),
                           rho_lv, c_eta*np.log(1/delta), p) for v in X]
    estimates += [get_mean(v[n//2:], np.mean(v[:n//2]),
                           rho_exp, c_eta*np.log(1/delta), p) for v in X]

    methods += n_trials*['mean'] + n_trials*['tm'] + n_trials * \
        ['mom'] + n_trials*['mean atm'] + n_trials * \
        ['mean win'] + n_trials*['mean lv']
    methods += n_trials*['mean exp']
    ns += 7*n_trials*[n]
    deltas += 7*n_trials*[delta]
    distributions += 7*n_trials*[distribution]
    c_etas += 7*n_trials*[c_eta]
    estimates += [get_mean(v[n//2:], get_mom(v[:n//2], int(np.ceil(c_eta*np.log(1/delta)))),
                           rho_atm, c_eta*np.log(1/delta), 1) for v in X]
    estimates += [get_mean(v[n//2:], get_mom(v[:n//2], int(np.ceil(c_eta*np.log(1/delta)))),
                           rho_win, c_eta*np.log(1/delta), 1) for v in X]

    p = 2*tail - 0.01 if tail <= 1 else 2
    estimates += [get_mean(v[n//2:], get_mom(v[:n//2], int(np.ceil(c_eta*np.log(1/delta)))),
                           rho_lv, c_eta*np.log(1/delta), p) for v in X]
    estimates += [get_mean(v[n//2:], get_mom(v[:n//2], int(np.ceil(c_eta*np.log(1/delta)))),
                           rho_exp, c_eta*np.log(1/delta), p) for v in X]
    methods += n_trials*['mom atm'] + n_trials * \
        ['mom win'] + n_trials*['mom lv']
    methods += n_trials*['mom exp']
    ns += 4*n_trials*[n]
    deltas += 4*n_trials*[delta]
    distributions += 4*n_trials*[distribution]
    c_etas += 4*n_trials*[c_eta]

    estimates += [get_mean(v[n//2:], get_tm(v[:n//2], int(np.ceil(c_eta*np.log(1/delta)))),
                           rho_atm, c_eta*np.log(1/delta), 1) for v in X]
    estimates += [get_mean(v[n//2:], get_tm(v[:n//2], int(np.ceil(c_eta*np.log(1/delta)))),
                           rho_win, c_eta*np.log(1/delta), 1) for v in X]

    p = 2*tail - 0.01 if tail <= 1 else 2
    estimates += [get_mean(v[n//2:], get_tm(v[:n//2], int(np.ceil(c_eta*np.log(1/delta)))),
                           rho_lv, c_eta*np.log(1/delta), p) for v in X]
    estimates += [get_mean(v[n//2:], get_tm(v[:n//2], int(np.ceil(c_eta*np.log(1/delta)))),
                           rho_exp, c_eta*np.log(1/delta), p) for v in X]
    methods += n_trials*['tm atm'] + n_trials * \
        ['tm win'] + n_trials*['tm lv']
    methods += n_trials*['tm exp']
    ns += 4*n_trials*[n]
    deltas += 4*n_trials*[delta]
    distributions += 4*n_trials*[distribution]
    c_etas += 4*n_trials*[c_eta]
    return ns, methods, estimates, deltas, distributions, c_etas

# range_ns = [n for n in range(50,500,25)]
# range_deltas = np.exp(-np.linspace(1,6,24))
# range_ts = [(np.inf, 0), (np.inf, .9), (.605, 0), (.605, .9), (1.005, 0), (1.005, .9)]
# range_distributions = ["Gaussian", "Skewed Gaussian", "St df=1.21", "Skewed St df=1.21", "St df=2.01",  "Skewed St df=2.01"]


range_ns = [100, 500]
range_deltas = [.1, .01, .001]
range_ts = [(np.inf, 0), (1.005, 0)]
range_distributions = ["Gaussian", "St df=2.01"]
range_c_eta = [0.5, 1, 2]
# range_ts = [(np.inf, 0.9), (1.005, 0.9), (1.005, 0.95), (1.005, 0.99) ]
# range_distributions = ["Skewed Gaussian", "St r=0.9", "St r=0.95", "St r=0.99"]

loop_n = len(range_ns)*len(range_deltas)*len(range_ts)
ret = Parallel(n_jobs=120)(delayed(task)(params) for params in tqdm(product(
    range_ns, range_deltas, range_c_eta, zip(range_ts, range_distributions)), total=loop_n))

df = [pd.DataFrame({'n': r[0], 'method': r[1], 'estimate': r[2],
                   'delta': r[3], 'distribution': r[4], "c_eta": r[5]}) for r in ret]
df = pd.concat(df)
df.to_parquet("results_bernoulli_symmetrical_split.parquet")
