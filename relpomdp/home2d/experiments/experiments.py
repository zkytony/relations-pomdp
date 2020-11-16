# Run experiments in test domains

from sciex import Experiment, Trial, Event, Result,\
    YamlResult, PklResult, PostProcessingResult
import argparse
from relpomdp.home2d.tests.test_pomdp_nk import test_pomdp_nk
from relpomdp.home2d.tests.test_pomdp import test_pomdp
from relpomdp.home2d.tests.test_mdp import test_mdp
from relpomdp.home2d.planning.test_subgoals_nk import test_subgoals_agent, subgoal_sequence
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
                trials_per_env=1,
                difficulty_threshold="Kitchen",
                target_classes={"Salt"}):
    env_path = os.path.join(FILE_PATHS["exp_worlds"], env_file)
    with open(env_path, "rb") as f:
        envs = pickle.load(f)

    with open(os.path.join(FILE_PATHS["exp_config"], domain_config_file)) as f:
        domain_config = yaml.load(f, Loader=yaml.Loader)

    # Read the scoring files
    df_corr = pd.read_csv(os.path.join(FILE_PATHS["exp_data"], corr_score_file))
    df_dffc = pd.read_csv(os.path.join(FILE_PATHS["exp_data"], dffc_score_file))
    df_subgoal = pd.read_csv(os.path.join(FILE_PATHS["exp_data"], subgoal_score_file))

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
            "num_sims": 2000
        }
    }

    all_trials = []
    for target_class in target_classes:

        # Get the room as difficult threshold
        for room_name in domain_config["objects"]:
            if target_class in domain_config["objects"][room_name]:
                difficulty_threshold = room_name
                break

        count = 0
        items = list(envs.items())
        random.shuffle(items)
        for env_id, env in items:
            shared_config["env_id"] = env_id
            # We will do:
            agent_types = {
                "pomdp-nk", "pomdp-subgoal-nk",
                "pomdp-subgoal-nk-nocorr",
                "random-nk", "heuristic-nk"
            }
            shared_config["target_class"] = target_class
            if len(env.ids_for(target_class)) == 0:
                continue

            subgoals = subgoal_sequence(target_class, df_subgoal, df_dffc,
                                        difficulty_threshold=difficulty_threshold)
            shared_config["subgoal_sequence"] = subgoals

            for agent_type in agent_types:
                config = copy.deepcopy(shared_config)
                config["agent_type"] = agent_type
                if "subgoal" in agent_type:
                    config["planning"]["difficulty_threshold"] = difficulty_threshold

                for i in range(trials_per_env):
                    trial_name = "search-%s-w%d-l%d-nrooms%d-nsubgoals%d_%d%d_%s"\
                        % (target_class.replace("-","#"), domain_config["width"], domain_config["length"],
                           domain_config["nrooms"], len(subgoals),
                           env_id, i, agent_type)
                    trial = RelPOMDPTrial(trial_name, config, verbose=True)
                    all_trials.append(trial)
            count += 1
            if count >= num_envs:\
                break

    random.shuffle(all_trials)
    output_dir = "./results"
    exp = Experiment("Search2DExperimentDD_%d-%d-nrooms%d" % (domain_config["width"], domain_config["length"], domain_config["nrooms"]),
                     all_trials, output_dir, verbose=True, add_timestamp=True)
    exp.generate_trial_scripts(split=6, exist_ok=False)
    print("Find multiple computers to run these experiments.")


if __name__ == "__main__":
    # Experiment-level configurations
    dims = (10, 10)
    if dims == (10, 10):
        env_file = "test-envs-10x10.pkl"
        domain_config_file = "10x10_10-20-2020.yaml"
    elif dims == (6, 6):
        env_file = "test-envs-6x6.pkl"
        domain_config_file = "6x6_10-31-2020.yaml"

    dffc_score_file = "difficulty-train-envs-6x6-pomdp-20201113134230180.csv"
    corr_score_file = "correlation_train-envs-6x6_20201113135539885.csv"
    subgoal_score_file = "scores_C#6x6_10-31-2020_D#train-envs-6x6-pomdp_20201113140449444.csv"

    target_classes = {"Salt", # Hard to detect, 1 subgoal
                      "Pepper", # Hard to detect, 2 subgoals
                      "Single-bed", # Easy to detect, 1 subgoals
                      "Computer"} # Easy to detect, 2 subgoals

    num_envs = 15
    trials_per_env = 1

    print("Are you sure you want to use")
    print("    Difficulty file: %s" % dffc_score_file)
    print("    Correlation file: %s" % corr_score_file)
    print("    Subgoal score file: %s" % subgoal_score_file)
    print("for %dx%d environments coming from '%s'" % (*dims, env_file))
    print("    with config: %s" % domain_config_file)
    answer = input("? [y] ")
    if answer.lower().startswith("y"):
        make_trials(env_file,
                    domain_config_file,
                    dffc_score_file,
                    corr_score_file,
                    subgoal_score_file,
                    num_envs=num_envs,
                    trials_per_env=trials_per_env,
                    target_classes=target_classes)
    else:
        print("Nothing will be generated.")
