from Experiment import Experiment
import numpy as np

# is shrinkage good?
exp1 = Experiment(
    name="is_shrinkage_good",
    base_estimators=["mean", "median", "mom", "tm"],
    shrinkage_function=[None, "lv", "atm", "win", "exp", "inv"],
    symmetrized=[False],
    normalized=[False],
    deltas=[5e-2],
    c_etas=[1.0],
    splits=[.05, .1, 0.25, 0.5, 0.75, 0.9, 0.95],
    dist_a=[1.005, np.inf],
    dist_r=[0.0, 0.99],
    ns=[100, 250, 500, 750, 1000],
    contamination_level=[0.0],
    n_trials=10000,
    n_jobs=20
)
exp1.run()
exp1.compute_metrics()
# exp2 = Experiment(
#     name="is_shrinkage_good",
#     base_estimators=["mean", "median", "mom", "tm"],
#     shrinkage_function=[None, "lv", "atm", "win", "exp", "inv"],
#     symmetrized=[False],
#     normalized=[False],
#     deltas=[5e-2],
#     c_etas=[1.0],
#     splits=[.05, .1, 0.25, 0.5, 0.75, 0.9, 0.95],
#     dist_a=[1.005, np.inf],
#     dist_r=[0.0, 0.99],
#     ns=[100, 250, 500, 750, 1000],
#     contamination_level=[0.0],
#     n_trials=1000,
#     n_jobs=20
# )
# exp2.run()
# exp2.compute_metrics()
