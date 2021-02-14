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
from baselines import *
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


# Making trials for experiments
def EXPERIMENT_varynobj(split=8, num_trials=NUM_TRIALS):
    """
    2. Fix size, randomize detector noise, vary the number of objects

    Will test 2, 3, 4, 5 objects on 5x5 domain.
    """
    # Experiment name
    exp_name = "Field2D-VaryingNumObj-5x5"
    start_time_str = dt.now().strftime("%Y%m%d%H%M%S%f")[:-3]
    exp_name += "_" + start_time_str

    # Build spec
    spec = {"name": "grid2d_2obj"}
    target_obj = (1, "blue-cube", [30, 30, 200])
    robot = (0, "robot", [10, 190, 10])
    add_object(spec, *target_obj, dim=[1,1])
    add_object(spec, *robot, dim=[1,1])

    # add target and target detector (0.8 true positive)
    target_id = target_obj[0]
    set_target(spec, target_id)
    blue_dspec = add_detector(spec, "blue-detector", 100, "loc", energy_cost=0.0)
    add_disk_sensor(blue_dspec, target_id, radius=0, true_positive=0.8)

    # add robot
    add_robot_simple2d(spec)

    # Set dimension
    set_dim(spec_, [5,5])

    # Creating Trials
    ## Deterministic random seeds
    rnd = random.Random(100)
    seeds = rnd.sample([i for i in range(1000, 10000)], num_trials)

    all_trials = []
    NOBJS = [2, 3, 4, 5]
    for nobj in NOBJS:
        print("case {}".format(nobj))
        spec_ = copy.deepcopy(spec)

        # Add nobj-1 additional objects.
        for objtup in OBJECTS[1:nobj]:
            assert objtup[0] != target_id
            add_object(spec_, *objtup, dim=[1,1])

            # Add


        for seed in seeds:
            pass






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

            # NO NEED TO PRUNE, because there are only two objects
            # Correlation heuristic planner. (No Pruning)
            all_trials.append(corr_heuristic_pouct_trial(spec_corr,
                                                         joint_dist_path, seed,
                                                         name_prefix, k=-1,
                                                         init_qvalue_lower_bound=True))


    random.shuffle(all_trials)
    exp = Experiment(exp_name, all_trials, OUTPUT_DIR, verbose=True,
                     add_timestamp=False)
    exp.generate_trial_scripts(split=split)
    print("Trials generated at %s/%s" % (exp._outdir, exp.name))
    print("Find multiple computers to run these experiments.")
