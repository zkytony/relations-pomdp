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

# Use relative paths
OUTPUT_DIR = os.path.join("results", "field2d")
RESOURCE_DIR = os.path.join("resources", "field2d")

# Shared configurations
NUM_TRIALS = 150  # number of trials for each data point

# Making trials for experiments
def EXPERIMENT_varynobj(split=8, num_trials=NUM_TRIALS):
    """
    2. Fix size, randomize detector noise, vary the number of objects

    Will test 2, 3, 4, 5 objects on 5x5 domain.
    """
    # Experiment name. Experiment will store at OUTPUT_DIR/exp_name
    dim = [5,5]
    exp_name = "Field2D-VaryingNumObj-{}x{}".format(dim[0], dim[1])
    start_time_str = dt.now().strftime("%Y%m%d%H%M%S%f")[:-3]
    exp_name += "_" + start_time_str

    # Build spec
    spec = {"name": "grid2d_2obj"}
    target_obj = (1, "blue-cube", [30, 30, 200])
    robot = (0, "robot", [10, 190, 10])
    add_object(spec, *target_obj, dim=[1,1])
    add_object(spec, *robot, dim=[1,1])

    # add target and target detector (SENSOR NOT ADDED)
    target_id = target_obj[0]
    set_target(spec, target_id)
    add_detector(spec, "blue-detector", 100, "loc", energy_cost=0.0)

    # add robot
    add_robot_simple2d(spec)

    # Set dimension
    set_dim(spec, dim)

    # Creating Trials
    ## Deterministic random seeds
    rnd = random.Random(100)
    seeds = rnd.sample([i for i in range(1000, 10000)], num_trials)

    all_trials = []
    NOBJS = [2, 3, 4, 5]
    for nobj in NOBJS:
        print("case {} objects".format(nobj))
        spec_ = copy.deepcopy(spec)

        for objtup in OBJECTS[1:nobj]:
            objid = objtup[0]

            # even objects nearby radius 2. Odd nearby radius 2
            if objid % 2 == 0:
                near_radius = 1
            else:
                near_radius = 2

            assert objid != target_id
            add_object(spec_, *objtup, dim=[1,1])

            # Add detector and sensor for the object (NO SENSOR YET)
            detector_name = "{}-detector".format(objtup[1].split("-")[0])
            detector_id = detid(objid)
            dspec_ = add_detector(spec_, detector_name, detector_id, "loc",
                                  energy_cost=0.0)

            # Add factor with respect to the target. Also, randomize the radius.
            add_factor(spec_, objects=[objid], dist_type="uniform")
            add_factor(spec_, objects=[objid, target_id], dist_type="nearby",
                       params={"radius": near_radius})

        # We parse the domain file once PER NOBJ amount, parse the joint distribution,
        # and then specify the path to that problem .pkl file.
        problem = problem_parser(spec_)
        os.makedirs(os.path.join(OUTPUT_DIR, exp_name, "resources"), exist_ok=True)
        joint_dist_file = "joint_dist_{}_{}obj.pkl".format(",".join(map(str,dim)), nobj)
        with open(os.path.join(OUTPUT_DIR, exp_name, "resources", joint_dist_file), "wb") as f:
            pickle.dump(problem.joint_dist, f)
        # Relative path to resources, with respect to experiment root
        joint_dist_path = os.path.join("resources", joint_dist_file)

        for seed in seeds:
            # Add nobj-1 additional objects.
            # Randomize the true positive rates per trial
            rnd = random.Random(seed)

            blue_dspec_ = get_detector(spec_, 100)
            add_disk_sensor(blue_dspec_, target_id, radius=0,
                            true_positive=rnd.uniform(0.8, 0.9))

            # Add sensors
            for objtup in OBJECTS[1:nobj]:
                objid = objtup[0]
                assert objid != target_id
                add_object(spec_, *objtup, dim=[1,1])
                dspec_ = get_detector(spec_, detid(objid))
                add_disk_sensor(dspec_, objid, radius=rnd.randint(1,2),
                                true_positive=rnd.uniform(0.9, 1.0))

            # Create two specs. One for the agent that uses correlation
            # and one that does not.
            spec_corr = copy.deepcopy(spec_)
            spec_targetonly = copy.deepcopy(spec_)
            for objtup in OBJECTS[1:nobj]:
                remove_detector(spec_targetonly, detid(objtup[0]))  # remove detector for the other object

            name_prefix = "varynobj-{}_{}".format(nobj, seed)

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

            # Correlation heuristic planner. (Pruned. k=2)
            all_trials.append(corr_heuristic_pouct_trial(spec_corr,
                                                         joint_dist_path, seed,
                                                         name_prefix, k=2,
                                                         init_qvalue_lower_bound=True))

            # Correlation heuristic planner. (Pruned. k=2, Not using qvalue init by lower bound)
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
    EXPERIMENT_varynobj(split=5, num_trials=3)
