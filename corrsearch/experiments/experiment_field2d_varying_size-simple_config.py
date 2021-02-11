"""
For the field2d domain. Vary the dimension of the search space.
Compare between two baselines: One leverages the correlation
for planning (i.e. considers the dependency), and one that does not.
"""

from corrsearch.experiments.domains.field2d.test_trial import make_config, make_trial
from corrsearch.experiments.domains.field2d.parser import *
import copy
import os
import random
import pickle
from sciex import Experiment
from datetime import datetime as dt

ABS_PATH = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(ABS_PATH, "results", "field2d")
RESOURCE_DIR = os.path.join(ABS_PATH, "resources", "field2d")
# True positive settings for the target detector
DIMS = [(3,3), (4,4), (5,5), (6,6)]

# general configs
PLANNER_CONFIG = {
    "max_depth": 25,
    "discount_factor": 0.95,
    "num_sims": 1500,
    "exploration_const": 200
}
MAX_STEPS = 30

# we want the same seeds every time
NUM_TRIALS_PER_SETTING = 30


def build_trials(exp_name):
    domain_file = "./domains/field2d/configs/simple_config.yaml"
    spec_original = read_domain_file(domain_file)

    # We parse the domain file once, parse the joint distribution,
    # and then specify the path to that problem .pkl file.
    problem = problem_from_file(domain_file)
    os.makedirs(os.path.join(RESOURCE_DIR, exp_name), exist_ok=True)
    joint_dist_path = os.path.join(RESOURCE_DIR, exp_name, "joint_dist.pkl")
    with open(joint_dist_path, "wb") as f:
        pickle.dump(problem.joint_dist, f)

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

        for seed in seeds:
            # seed for world generation
            baseline = "corr"
            config = make_config(spec_corr_agent, init_locs="random",
                                 joint_dist_path=joint_dist_path,
                                 seed=seed, init_belief="prior",
                                 planner_config=PLANNER_CONFIG,
                                 max_steps=MAX_STEPS, visualize=False)
            trial_name = "varysize-{}_{}_{}".format(",".join(dim), seed, baseline)
            all_trials.append(make_trial(config, trial_name=trial_name))

            baseline = "target-only"
            config = make_config(spec_targetonly_agent, init_locs="random",
                                 joint_dist_path=joint_dist_path,
                                 seed=seed, init_belief="uniform",
                                 planner_config=PLANNER_CONFIG,
                                 max_steps=MAX_STEPS, visualize=False)
            trial_name = "varysize-{}_{}_{}".format(",".join(dim), seed, baseline)
            all_trials.append(make_trial(config, trial_name=trial_name))
    return all_trials


if __name__ == "__main__":
    # Experiment name
    exp_name = "Field2D-VaryingSize-SimpleConfig"
    start_time_str = dt.now().strftime("%Y%m%d%H%M%S%f")[:-3]
    exp_name += "_" + start_time_str

    # build trials
    trials = build_trials(exp_name)
    random.shuffle(trials)
    exp = Experiment(exp_name,
                     trials, OUTPUT_DIR,
                     verbose=True, add_timestamp=False)

    exp.generate_trial_scripts(split=8)
    print("Trials generated at %s/%s" % (exp._outdir, exp.name))
    print("Find multiple computers to run these experiments.")
