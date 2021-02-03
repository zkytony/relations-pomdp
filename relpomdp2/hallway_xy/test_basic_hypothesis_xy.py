"""This script tests the hypothesis that
an agent with the ability to perceive correlated
object should have a policy that obtains better
reward because of its ability to use that perception
ability.

This is for the two object case.
"""

import pomdp_py
import os, sys
import random
from datetime import datetime
import numpy as np
from hallway_xy import *
from hallway_xy import main as run_domain_xy
# from hallway_xy_joint import main as run_domain_xy_joint
from sciex import Trial, MainFuncTrial
import pickle
import pandas as pd
import seaborn as sns

# hallway length
HALLWAY_LEN = 8
SPATIAL_CORR_FUNC = spatially_close
RANGE_X = 0
RANGE_Y = 1
NTRIALS = 100
ACTIONS_ONLY_TARGET =  [HSAction("left"), HSAction("right"), HSAction("Declare"), HSAction("lookX")]
ACTIONS_BOTH_DETECTABLE = [HSAction("left"), HSAction("right"), HSAction("Declare"), HSAction("lookX"), HSAction("lookY")]
NUM_STEPS = 100

suffix = "%d_%s_%d_%d" % (HALLWAY_LEN, SPATIAL_CORR_FUNC.__name__, RANGE_X, RANGE_Y)

policies_filename = "policies_%s.pkl" % suffix
if os.path.exists(policies_filename):
    with open(policies_filename, "rb") as f:
        policies_only_target, policies_both_detectable = pickle.load(f)
else:
    # Obtain the policy for each initial robot position
    policies_both_detectable = {}
    policies_only_target = {}
    for r in range(HALLWAY_LEN):
         setting = random_setting(HALLWAY_LEN, init_r=r)

         common_args = {"spatial_corr_func": SPATIAL_CORR_FUNC,
                        "range_x": RANGE_X,
                        "range_y": RANGE_Y,
                        "setting": setting,
                        "viz": False}

         # only target detectable
         res = run_domain_xy(solver="sarsop",
                             actions=ACTIONS_ONLY_TARGET,
                             **common_args)
         policies_only_target[r] = res[0]

         # both detectable
         res = run_domain_xy(solver="sarsop",
                             actions=ACTIONS_BOTH_DETECTABLE,
                             **common_args)
         policies_both_detectable[r] = res[0]

with open(policies_filename, "wb") as f:
     pickle.dump((policies_only_target, policies_both_detectable), f)


data = []
for t in range(NTRIALS):
    locations = set(i for i in range(HALLWAY_LEN))
    init_r = random.sample(locations, 1)[0]
    setting = random_setting(HALLWAY_LEN, init_r=init_r, spatial_corr_func=SPATIAL_CORR_FUNC)

    init_args = {"spatial_corr_func": SPATIAL_CORR_FUNC,
                   "range_x": RANGE_X,
                   "range_y": RANGE_Y}
    sim_args = {"discount_factor": 0.95,
                "num_steps": NUM_STEPS,
                "viz": False}

    # ONLY TARGET
    policy = policies_only_target[init_r]
    agent, env = initialize_setting(setting,
                                    actions=ACTIONS_ONLY_TARGET,
                                    **init_args)
    value = policy.value(agent.belief)
    _, _, _, cum_reward = simulate_policy(policy, agent, env,
                                          **sim_args)
    data.append([t, setting, "X", cum_reward, value])

    # BOTH OBJECTS
    policy = policies_both_detectable[init_r]
    agent, env = initialize_setting(setting,
                                    actions=ACTIONS_BOTH_DETECTABLE,
                                    **init_args)
    value = policy.value(agent.belief)
    _, _, _, cum_reward = simulate_policy(policy, agent, env,
                                          **sim_args)
    data.append([t, setting, "XY", cum_reward, value])

df = pd.DataFrame(data, columns=["trial_no", "setting", "Detectable", "return", "Vb0"])
timestamp = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
df.to_csv("results_%s_%s.csv" % (suffix, timestamp))

print(df.groupby("Detectable").agg(["mean","std"]))
