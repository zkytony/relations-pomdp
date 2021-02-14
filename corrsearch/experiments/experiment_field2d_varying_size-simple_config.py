"""
For the field2d domain. Vary the dimension of the search space.
Compare between two baselines: One leverages the correlation
for planning (i.e. considers the dependency), and one that does not.

This script is for preliminary experiments.
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
# True positive settings for the target detector
DIMS = [(2,2), (3,3), (4,4), (5,5)]

# general configs
MAX_STEPS = 200

# we want the same seeds every time
NUM_TRIALS_PER_SETTING = 30


def build_trials(exp_name, config_name):
    domain_file = "./domains/field2d/configs/{}.yaml".format(config_name)
    spec_original = read_domain_file(domain_file)

    rnd = random.Random(100)
    seeds = [rnd.randint(1000, 10000)
             for i in range(NUM_TRIALS_PER_SETTING)]

    all_trials = []

    for dim in DIMS:
        print("case {}".format(dim))
        spec = copy.deepcopy(spec_original)

        # set the dimension
        spec["dim"] = list(dim)

        # Create two specs. One for the agent that uses correlation
        # and one that does not.
        spec_corr_agent = copy.deepcopy(spec)
        spec_targetonly_agent = copy.deepcopy(spec)
        dpsec_popped = spec_targetonly_agent["detectors"].pop(1)
        assert dpsec_popped["name"] == "red-detector"

        # We parse the domain file once PER DOMAIN SIZE, parse the joint distribution,
        # and then specify the path to that problem .pkl file.
        problem = problem_parser(spec)
        os.makedirs(os.path.join(RESOURCE_DIR, exp_name), exist_ok=True)
        joint_dist_path = os.path.join(RESOURCE_DIR, exp_name,
                                       "joint_dist_{}.pkl".format(",".join(map(str,dim))))
        with open(joint_dist_path, "wb") as f:
            pickle.dump(problem.joint_dist, f)


        for seed in seeds:
            # seed for world generation
            name_prefix = "varysize-{}_{}".format(",".join(map(str,dim)), seed)

            # Correlation used. POMCP Planning
            baseline = "corr"
            config = make_config(spec_corr_agent, init_locs="random",
                                 joint_dist_path=joint_dist_path,
                                 seed=seed, init_belief="prior",
                                 planner="pomdp_py.POUCT",
                                 planner_config=POMCP_PLANNER_CONFIG,
                                 max_steps=MAX_STEPS, visualize=False)
            trial_name = "{}_{}".format(name_prefix, baseline)
            all_trials.append(make_trial(config, trial_name=trial_name))

            # Correlation not used. POMCP Planning (ablation)
            baseline = "target-only"
            config = make_config(spec_targetonly_agent, init_locs="random",
                                 joint_dist_path=joint_dist_path,
                                 seed=seed, init_belief="uniform",
                                 planner="pomdp_py.POUCT",
                                 planner_config=POMCP_PLANNER_CONFIG,
                                 max_steps=MAX_STEPS, visualize=False)
            trial_name = "{}_{}".format(name_prefix, baseline)
            all_trials.append(make_trial(config, trial_name=trial_name))

            # Correlation used. Entropy minimization (baseline)
            baseline = "entropymin"
            config = make_config(spec_corr_agent, init_locs="random",
                                 joint_dist_path=joint_dist_path,
                                 seed=seed, init_belief="prior",
                                 planner="EntropyMinimizationPlanner",
                                 planner_config=ENTROPY_PLANNER_CONFIG,
                                 max_steps=MAX_STEPS, visualize=False)
            trial_name = "{}_{}".format(name_prefix, baseline)
            all_trials.append(make_trial(config, trial_name=trial_name))

            # Correlation used. Random Planner (baseline)
            baseline = "random"
            config = make_config(spec_corr_agent, init_locs="random",
                                 joint_dist_path=joint_dist_path,
                                 seed=seed, init_belief="prior",
                                 planner="RandomPlanner",
                                 planner_config=RANDOM_PLANNER_CONFIG,
                                 max_steps=MAX_STEPS, visualize=False)
            trial_name = "{}_{}".format(name_prefix, baseline)
            all_trials.append(make_trial(config, trial_name=trial_name))

            # Correlation used. Heuristic sequential
            baseline = "heuristic"
            config = make_config(spec_corr_agent, init_locs="random",
                                 joint_dist_path=joint_dist_path,
                                 seed=seed, init_belief="prior",
                                 planner="HeuristicSequentialPlanner",
                                 planner_config=HEURISTIC_ONLINE_PLANNER_CONFIG,
                                 max_steps=MAX_STEPS, visualize=False)
            trial_name = "{}_{}".format(name_prefix, baseline)
            all_trials.append(make_trial(config, trial_name=trial_name))


    return all_trials


if __name__ == "__main__":
    # Experiment name
    config_name = "simple_config"
    exp_name = "Field2D-VaryingSize-{}".format(config_name.title())
    start_time_str = dt.now().strftime("%Y%m%d%H%M%S%f")[:-3]
    exp_name += "_" + start_time_str

    # build trials
    trials = build_trials(exp_name, config_name)
    random.shuffle(trials)
    exp = Experiment(exp_name,
                     trials, OUTPUT_DIR,
                     verbose=True, add_timestamp=False)

    exp.generate_trial_scripts(split=8)
    print("Trials generated at %s/%s" % (exp._outdir, exp.name))
    print("Find multiple computers to run these experiments.")
