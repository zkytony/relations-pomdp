"""
Generating experiment scripts for the field2d domain.

Three experiments:
1. Fix number of objects, fix detector noise, vary the size of search environment
2. Fix size, randomize detector noise, vary the number of objects
3. Fix number of objects, size, vary target detector noise, randomize other noises.

Baselines:
1. Random planner
2. Entropy minimization planner
3. Target-only agent
4. Correlation agent
5. Correlation + Heuristic (no pruning)
6. Correlation + Heuristic (pruning, k=2)

This is the script that generated the experiment trials for the paper.
"""

from corrsearch.experiments.domains.field2d.test_trial import make_config, make_trial
from corrsearch.experiments.domains.field2d.parser import *
import copy
import random
from defaults import *

MAX_STEPS = 200   # maximum number of search steps

# Functions to create a trial for a baseline
def random_trial(spec,
                 joint_dist_path,  # required
                 seed,             # required
                 name_prefix):     # prefix for the trial name (experiment global config)
    assert joint_dist_path is not None, "Must supply joint distribution path"
    assert seed is not None, "Must supply seed"
    baseline = "random"
    config = make_config(spec, planner="RandomPlanner",
                         planner_config=RANDOM_PLANNER_CONFIG,
                         init_locs="random", init_belief="prior",
                         joint_dist_path=joint_dist_path, seed=seed,
                         max_steps=MAX_STEPS, visualize=False)
    trial_name = "{}_{}".format(name_prefix, baseline)
    return make_trial(config, trial_name=trial_name)

def entropymin_trial(spec, joint_dist_path, seed, name_prefix):
    assert joint_dist_path is not None, "Must supply joint distribution path"
    assert seed is not None, "Must supply seed"
    baseline = "entropymin"
    config = make_config(spec, planner="EntropyMinimizationPlanner",
                         planner_config=ENTROPY_PLANNER_CONFIG,
                         init_locs="random", init_belief="prior",
                         joint_dist_path=joint_dist_path, seed=seed,
                         max_steps=MAX_STEPS, visualize=False)
    trial_name = "{}_{}".format(name_prefix, baseline)
    return make_trial(config, trial_name=trial_name)


def target_only_pouct_trial(spec, joint_dist_path, seed, name_prefix):
    """Uses POUCT with random rollout"""
    assert joint_dist_path is not None, "Must supply joint distribution path"
    assert seed is not None, "Must supply seed"

    # Make sure there is only one detector, for the target
    assert len(spec["detectors"]) == 1, "Target only baseline uses only one detector"
    target_id = spec["target_id"]
    assert target_id in spec["detectors"][0]["sensors"],\
        "Target must be detectable by the target detector"

    baseline = "target-only-pouct"
    config = make_config(spec, planner="pomdp_py.POUCT",
                         planner_config=POMCP_PLANNER_CONFIG,
                         init_locs="random", init_belief="uniform",
                         joint_dist_path=joint_dist_path, seed=seed,
                         max_steps=MAX_STEPS, visualize=False)
    trial_name = "{}_{}".format(name_prefix, baseline)
    return make_trial(config, trial_name=trial_name)


def corr_pouct_trial(spec,
                     joint_dist_path,
                     seed,
                     name_prefix):
    """Uses POUCT with random rollout"""
    assert joint_dist_path is not None, "Must supply joint distribution path"
    assert seed is not None, "Must supply seed"

    # Make sure there is only one detector, for the target
    assert len(spec["detectors"]) > 1, "Expecting mulitple detectors for corr_pouct baseline"

    baseline = "corr-pouct"
    config = make_config(spec, planner="pomdp_py.POUCT",
                         planner_config=POMCP_PLANNER_CONFIG,
                         init_locs="random", init_belief="prior",
                         joint_dist_path=joint_dist_path, seed=seed,
                         max_steps=MAX_STEPS, visualize=False)
    trial_name = "{}_{}".format(name_prefix, baseline)
    return make_trial(config, trial_name=trial_name)

def corr_heuristic_pouct_trial(spec, joint_dist_path, seed, name_prefix, k=-1,
                               init_qvalue_lower_bound=True):
    """Uses HeuristicSequentialPlanner"""
    assert joint_dist_path is not None, "Must supply joint distribution path"
    assert seed is not None, "Must supply seed"

    # Make sure there is only one detector, for the target
    assert len(spec["detectors"]) > 1, "Expecting mulitple detectors for corr_pouct baseline"

    # baseline naming
    pruning = "noprune" if k <= 0 else "k={}".format(k)
    init_qvals = "iq" if init_qvalue_lower_bound else "noiq"
    baseline = "heuristic#%s#%s" % (pruning, init_qvals)

    planner_config = copy.deepcopy(HEURISTIC_ONLINE_PLANNER_CONFIG)
    planner_config["k"] = k  # number of detectors to plan with
    planner_config["init_qvalue_lower_bound"] = init_qvalue_lower_bound

    config = make_config(spec, planner="HeuristicSequentialPlanner",
                         planner_config=planner_config, init_locs="random",
                         init_belief="prior", joint_dist_path=joint_dist_path,
                         seed=seed, max_steps=MAX_STEPS, visualize=False)
    trial_name = "{}_{}".format(name_prefix, baseline)
    return make_trial(config, trial_name=trial_name)
