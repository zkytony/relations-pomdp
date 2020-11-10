# Run experiments in test domains

from sciex import Experiment, Trial, Event, Result,\
    YamlResult, PklResult, PostProcessingResult
import argparse
from relpomdp.home2d.tests.test_pomdp_nk import test_pomdp_nk
from relpomdp.home2d.tests.test_pomdp import test_pomdp
from relpomdp.home2d.tests.test_mdp import test_mdp
from relpomdp.home2d.planning.test_subgoals_nk import test_subgoals_agent
from relpomdp.home2d.learning.generate_worlds import generate_world
from relpomdp.home2d.utils import save_images_and_compress, discounted_cumulative_reward
from relpomdp.home2d.experiments.trial import RelPOMDPTrial
from relpomdp.home2d.constants import FILE_PATHS
from datetime import datetime as dt
import pandas as pd
import yaml
import random
import copy
import pickle
import os
import copy


def make_trials(env_file,
                domain_config_file,
                dffc_score_file,
                corr_score_file,
                subgoal_score_file,
                num_envs=10,
                trials_per_env=1):
    env_path = os.path.join(FILE_PATHS["exp_worlds"], env_file)
    with open(env_path, "rb") as f:
        envs = pickle.load(f)

    with open(os.path.join(FILE_PATHS["exp_config"], domain_config_file)) as f:
        domain_config = yaml.load(f, Loader=yaml.Loader)

    shared_config = {
        "domain": domain_config,
        "env_file": env_file,
        "visualize": False,
        "corr_score_file": corr_score_file,
        "dffc_score_file": dffc_score_file,
        "subgoal_score_file": subgoal_score_file,
        "planning": {
            "discount_factor": 0.95,
            "nsteps": 100,
            "max_depth": 30,
            "num_sims": 1000
        }
    }

    all_trials = []
    count = 0
    for env_id in envs:
        env = envs[env_id]
        shared_config["env_id"] = env_id
        # We will do:
        agent_types = {
            "pomdp-nk", "pomdp-subgoal-nk",
            "pomdp-subgoal-nk-nocorr",
            "random-nk", "heuristic-nk"
        }
        target_class = "Salt"  # TODO: TRY MORE?
        shared_config["target_class"] = target_class
        if len(env.ids_for(target_class)) == 0:
            continue

        for agent_type in agent_types:
            config = copy.deepcopy(shared_config)
            config["agent_type"] = agent_type
            if "subgoal" in agent_type:
                config["planning"]["difficulty_threshold"] = "Kitchen"

            for i in range(trials_per_env):
                trial_name = "search-%s-%d-%d_%d%d_%s"\
                    % (target_class, domain_config["width"], domain_config["length"],
                       env_id, i, agent_type)
                trial = RelPOMDPTrial(trial_name, config, verbose=True)
                all_trials.append(trial)
        count += 1
        if count >= num_envs:\
            break

    random.shuffle(all_trials)
    output_dir = "./results"
    exp = Experiment("Search2DExperiment_%d-%d" % (domain_config["width"], domain_config["length"]),
                     all_trials, output_dir, verbose=True, add_timestamp=False)
    exp.generate_trial_scripts(split=6, exist_ok=True)
    print("Find multiple computers to run these experiments.")


if __name__ == "__main__":
    # Experiment-level configurations
    dims = (6, 6)
    if dims == (10, 10):
        env_file = "test-envs-10x10.pkl"
        domain_config_file = "10x10_10-20-2020.yaml"
    elif dims == (6, 6):
        env_file = "test-envs-6x6.pkl"
        domain_config_file = "6x6_10-31-2020.yaml"

    dffc_score_file = "difficulty-try1-10-20-2020-20201026162744897.csv"
    corr_score_file = "correlation-try1-10-20-2020.csv"
    subgoal_score_file = "subgoal-scores=try1.csv"

    num_envs = 10
    trials_per_env = 1

    make_trials(env_file,
                domain_config_file,
                dffc_score_file,
                corr_score_file,
                subgoal_score_file,
                num_envs=num_envs,
                trials_per_env=trials_per_env)
