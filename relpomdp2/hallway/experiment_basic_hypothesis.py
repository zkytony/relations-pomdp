"""Runs the test_basic_hypothesis script many times to do an experiment"""
from test_basic_hypothesis import RUN, chmap, fpfn, make_actions
from relpomdp2.hallway.problem import *
import pandas as pd
import os
from datetime import datetime

DEFAULT_PARAMS = dict(
    HALLWAY_LEN = 6,
    NOBJS = 2,
    SPATIAL_CORR_FUNC = spatially_close,
    SENSOR_RANGES = chmap([0, 1]),
    SENSOR_COSTS = chmap([0, 0]),
    SENSOR_NOISES = chmap([fpfn(0.0, 0.0),
                           fpfn(0.0, 0.0)]),
                           # fpfn(0.0, 0.0)])
    PAIRWISE_RELATIONS = {("A", "B"): spatially_close},
    ACTIONS_ONLY_TARGET = make_actions(["A"]),
    ACTIONS_BOTH_DETECTABLE = make_actions(["A", "B"]),
    NUM_STEPS = 100,
    NTRIALS = 100,
    DISCOUNT_FACTOR=0.95,
    SARSOP_MEMORY = 200, # MB
    SARSOP_TIME = 180, # seconds
    SARSOP_PRECISION=1e-12,
    USE_CORRELATION_PRIOR=True,
    DEBUGGING=False
)


# Perfect sensor, increasing size
def exp_perfect_sensor_increasing_size():
    print("~~~~~~~~~~~ EXPERIMENT: Perfect sensor, increasing size ~~~~~~~~~~~~~")
    sizes = [4, 6, 7, 8]  # hallway lengtsh
    rows = []
    for hwl in sizes:
        print("===== HALLWAY LEN: %d =====================" % hwl)
        params = dict(DEFAULT_PARAMS)
        params["HALLWAY_LEN"] = hwl
        df = RUN(**params)
        for index, row in df.iterrows():
            rows.append(["%d-%s" % (hwl, row[2]), hwl] + list(row))
    expdf = pd.DataFrame(rows, columns=["HWL-D", "hallway_length"] + list(df.columns.values))
    timestamp = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
    expdf.to_pickle(os.path.join("results",
                                 "EXP_perfect-sensor-increasing-size_%s.pkl" % timestamp))
    return expdf

# Same size, increasing noise
def exp_same_size_increasing_noise():
    print("~~~~~~~~~~~ EXPERIMENT: Same size, increasing noise  ~~~~~~~~~~~~~")
    noises = [(0.0, 0.0), (0.01, 0.01), (0.05, 0.05), (0.1, 0.1)]#, (0.2, 0.2)]
    rows = []
    for nl, (fp, fn) in enumerate(noises):
        print("===== NL %d; FP: %.3f; FN: %.3f =====================" % (nl, fp, fn))
        params = dict(DEFAULT_PARAMS)
        params["SENSOR_NOISES"] = chmap([fpfn(fp, fn),  # target object
                                         fpfn(0.0, 0.0)])   # other object
        df = RUN(**params)
        for index, row in df.iterrows():
            rows.append(["%s-%s" % (str(nl), row[2]), nl, fp, fn] + list(row))
    expdf = pd.DataFrame(rows, columns=["NL-D", "NL", "FP", "FN"] + list(df.columns.values))
    timestamp = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
    expdf.to_pickle(os.path.join("results",
                                 "EXP_same-size-increasing-noise_%s.pkl" % timestamp))
    return expdf

if __name__ == "__main__":
    exp1 = exp_perfect_sensor_increasing_size()
    exp2 = exp_same_size_increasing_noise()

    print("\n::::::::::::::: Results for exp_perfect_sensor_increasing_size :::::::::::::::")
    print(exp1.groupby("HWL-D").agg(["mean", "std"]))

    print("\n::::::::::::::: Results for exp_same_size_increasing_noise :::::::::::::::")
    print(exp2.groupby("NL-D").agg(["mean", "std"]))
