from Experiment import Experiment
import numpy as np

# Experiment.rng keys the samples on the distribution alone, so wherever these three
# experiments overlap -- the compliant shrinkage functions at n=500, the uncontaminated
# column -- they are computed on identical data and their tables agree exactly.

# Table 1 and Figure 1.
exp1 = Experiment(
    name="is_shrinkage_good",
    base_estimators=["mean", "median", "mom", "tm"],
    shrinkage_function=[None, "lv", "atm", "win", "exp", "inv", "inv_root",
                        "tukey", "andrews", "hampel", "tanh"],
    symmetrized=[False],
    normalized=[False],
    deltas=[5e-2],
    c_etas=[1.0],
    dist_a=[1.005, np.inf],
    dist_r=[0.0, 0.99],
    ns=[150, 250, 500, 750, 1000],
    contamination_level=[0.0],
    n_trials=10000,
    n_jobs=20
)
exp1.run()
exp1.compute_metrics()
# Table 2.
exp2 = Experiment(
    name="contamination_robustness",
    base_estimators=["mean", "median", "mom", "tm"],
    shrinkage_function=[None, "lv", "atm", "win", "exp", "inv", "inv_root",
                        "tukey", "andrews", "hampel", "tanh"],
    symmetrized=[False],
    normalized=[False],
    deltas=[5e-2],
    c_etas=[1.0],
    dist_a=[1.005, np.inf],
    dist_r=[0.0, 0.99],
    ns=[500],
    contamination_level=[0.0, 0.05, 0.1, 0.2],
    n_trials=10000,
    n_jobs=20,
)
exp2.run()
exp2.compute_metrics()
# Table 3. Repeats every shrinkage function of exp1 so that the assumption-violating ones
# are read against the full set of compliant ones, on the same samples.
exp3 = Experiment(
    name="bad_shrinkage",
    base_estimators=["mean", "median", "mom", "tm"],
    shrinkage_function=[None, "lv", "atm", "win", "exp", "inv", "inv_root",
                        "tukey", "andrews", "hampel", "tanh",
                        "ln", "ln_sq", "sqrt", "inv_sqrt"],
    symmetrized=[False],
    normalized=[False],
    deltas=[5e-2],
    c_etas=[1.0],
    dist_a=[1.005, np.inf],
    dist_r=[0.0, 0.99],
    ns=[500],
    contamination_level=[0.0],
    n_trials=10000,
    n_jobs=20
)
exp3.run()
exp3.compute_metrics()
