from Experiment import Experiment
import numpy as np

exp1 = Experiment(
    name="is_shrinkage_good",
    base_estimators=["mean", "median", "mom", "tm"],
    shrinkage_function=[None, "lv", "atm", "win", "exp", "inv"],
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
exp1_reduced = Experiment(
    name="is_shrinkage_good_reduced",
    base_estimators=["mean", "median", "mom", "tm"],
    shrinkage_function=[None, "win", "inv"],
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
exp1_reduced.run()
exp1_reduced.compute_metrics()
exp3 = Experiment(
    name="contamination_robustness",
    base_estimators=["mean", "median", "mom", "tm"],
    shrinkage_function=[None, "lv", "atm", "win", "exp", "inv"],
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
    seed=0,
)
exp3.run()
exp3.compute_metrics()
exp4 = Experiment(
    name="bad_shrinkage",
    base_estimators=["mean", "median", "mom", "tm"],
    shrinkage_function=[None, "win", "inv", "ln", "ln_sq", "sqrt", "inv_sqrt"],
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
exp4.run()
exp4.compute_metrics()
