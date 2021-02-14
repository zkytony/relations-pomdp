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
import os
import random
import pickle
from sciex import Experiment
from datetime import datetime as dt
from defaults import *

ABS_PATH = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(ABS_PATH, "results", "field2d")
RESOURCE_DIR = os.path.join(ABS_PATH, "resources", "field2d")

# Shared configurations
NUM_TRIALS = 150  # number of trials for each data point
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
                         init_locs="random", init_belief="prior",
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

# Making trials for experiments
def EXPERIMENT_varysize(split=8, num_trials=NUM_TRIALS):
    """
    Fix number of objects, randomize detector noise, vary the size of search environment
    We will do **2 objects**. Target detector TP=0.8~0.9. Other object detector is TP=0.9~1.0
    """
    # Experiment name
    exp_name = "Field2D-VaryingSize-2Obj"
    start_time_str = dt.now().strftime("%Y%m%d%H%M%S%f")[:-3]
    exp_name += "_" + start_time_str

    # Build spec
    spec = {"name": "grid2d_2obj"}
    target_obj = (1, "blue-cube", [30, 30, 200])
    other_obj = (2, "red-cube", [232, 55, 35])
    robot = (0, "robot", [10, 190, 10])
    add_object(spec, *target_obj, dim=[1,1])
    add_object(spec, *other_obj, dim=[1,1])
    add_object(spec, *robot, dim=[1,1])

    target_id = target_obj[0]
    set_target(spec, target_id)

    # add two detectors, one for each object
    blue_dspec = add_detector(spec, "blue-detector", 100, "loc", energy_cost=0.0)
    red_dspec = add_detector(spec, "red-detector", 200, "loc", energy_cost=0.0)
    # add disk sensors for each object. Informative object has better sensor
    add_disk_sensor(blue_dspec, target_id, radius=0, true_positive=0.8)
    add_disk_sensor(red_dspec, target_id, radius=1, true_positive=0.9)

    target_true_pos = random.uniform(0.8, 0.9)
    other_true_pos = random.uniform(0.9, 1.0)

    # add probability factors
    add_factor(spec, objects=[other_obj[0]], dist_type="uniform")
    add_factor(spec, classes=[target_obj[1], other_obj[1]], dist_type="nearby", params={"radius":1})

    # add robot
    add_robot_simple2d(spec)

    # Creating Trials
    ## Deterministic random seeds
    rnd = random.Random(100)
    seeds = rnd.sample([i for i in range(1000, 10000)], num_trials)

    all_trials = []
    DIMS = [(2,2), (3,3), (4,4), (5,5), (6,6)]
    for dim in DIMS:
        print("case {}".format(dim))

        spec_ = copy.deepcopy(spec) # for safety
        set_dim(spec_, dim)

        # We parse the domain file once PER DOMAIN SIZE, parse the joint distribution,
        # and then specify the path to that problem .pkl file.
        problem = problem_parser(spec_)
        os.makedirs(os.path.join(RESOURCE_DIR, exp_name), exist_ok=True)
        joint_dist_path = os.path.join(RESOURCE_DIR, exp_name,
                                       "joint_dist_{}.pkl".format(",".join(map(str,dim))))
        with open(joint_dist_path, "wb") as f:
            pickle.dump(problem.joint_dist, f)

        # Create two specs. One for the agent that uses correlation
        # and one that does not.
        spec_corr = copy.deepcopy(spec_)
        spec_targetonly = copy.deepcopy(spec_)
        remove_detector(spec_targetonly, 200)  # remove detector for the other object

        for seed in seeds:
            # seed for world generation
            name_prefix = "varysize-{}_{}".format(",".join(map(str,dim)), seed)

            # Random trial. Correlation used.
            all_trials.append(random_trial(spec_corr, joint_dist_path, seed,
                                           name_prefix))

            # Entropy minimization trial. Correlation used.
            all_trials.append(entropymin_trial(spec_corr, joint_dist_path, seed,
                                               name_prefix))

            # Target only POUCT. Correlation NOT USED (ablation)
            all_trials.append(target_only_pouct_trial(spec_targetonly,
                                                      joint_dist_path, seed, name_prefix))

            # Correlation used POUCT.
            all_trials.append(corr_pouct_trial(spec_corr, joint_dist_path, seed,
                                               name_prefix))

            # Correlation heuristic planner. (No Pruning)
            all_trials.append(corr_heuristic_pouct_trial(spec_corr,
                                                         joint_dist_path, seed,
                                                         name_prefix, k=-1,
                                                         init_qvalue_lower_bound=True))

            # Correlation heuristic planner. (Pruning k=2)
            all_trials.append(corr_heuristic_pouct_trial(spec_corr,
                                                         joint_dist_path, seed,
                                                         name_prefix, k=2,
                                                         init_qvalue_lower_bound=True))

            # Correlation heuristic planner. (Pruning k=2, no value initialization)
            all_trials.append(corr_heuristic_pouct_trial(spec_corr,
                                                         joint_dist_path, seed,
                                                         name_prefix, k=2,
                                                         init_qvalue_lower_bound=False))

    random.shuffle(all_trials)
    exp = Experiment(exp_name, all_trials, OUTPUT_DIR, verbose=True,
                     add_timestamp=False)
    exp.generate_trial_scripts(split=split)
    print("Trials generated at %s/%s" % (exp._outdir, exp.name))
    print("Find multiple computers to run these experiments.")


if __name__ == "__main__":
    EXPERIMENT_varysize(split=5, num_trials=3)
