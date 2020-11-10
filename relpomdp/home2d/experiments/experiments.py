# Run experiments in test domains
#
# We will test on 10x10 worlds (same size as training)
# We will randomly generate N worlds (~30). And record
# the cumulative discounted rewards of a couple of baselines.
# We also record the success rate of finding the goal object.
# (I'm sure these are not the eventual metrics, but good for now).

import argparse
from relpomdp.home2d.agent.tests.test_pomdp_nk import test_pomdp_nk
from relpomdp.home2d.agent.tests.test_pomdp import test_pomdp
from relpomdp.home2d.agent.tests.test_mdp import test_mdp
from relpomdp.home2d.learning.testing.test_subgoals_nk import test_subgoals_agent
from relpomdp.home2d.learning.generate_worlds import generate_world
from relpomdp.home2d.utils import save_images_and_compress, discounted_cumulative_reward
from relpomdp.home2d.learning.testing.test_utils import add_room_states
from relpomdp.home2d.experiments import RelPOMDPTrial
from datetime import datetime as dtnnnnnn
import pandas as pd
import yaml
import copy
import os
import copy


def make_trials(env_file,
                domain_config_file,
                dffc_score_file,
                corr_score_file,
                subgoal_score_file,
                num_envs=10,
                trials_per_env=1):
    env_path = os.path.join(FILE_PATHS["exp_worlds"], self._config["env_file"])
    with open(env_path, "rb") as f:
        envs = pickle.load(f)

    with open(os.path.join(FILE_PATHS["exp_configs"], domain_config_file)) as f:
        domain_config = yaml.load(f)

    shared_config = {
        "domain": domain_config,
        "env_file": env_file,
        "visualize": False,
        "corr_score_file": corr_score_file,
        "dffc_score_file": corr_score_file,
        "subgoal_score_file": corr_score_file,
        "planning": {
            "discount_factor": 0.95,
            "nsteps": 100,
            "max_depth": 30,
            "num_sims": 1000
        }
    }

    for env_id in envs:
        shared_config["env_id"] = env_id
        # We will do:
        agent_types = {
            "pomdp-nk", "pomdp-subgoal-nk",
            "pomdp-subgoal-nk-no_corr",
            "random-nk", "heuristic-nk"
        }
        target_class = "Salt"  # TODO: TRY MORE?
        for agent_type in agent_types:
            config = copy.deepcopy(shared_config)
            config["agent_type"] = agent_type
            if "subgoal" in agent_type:
                config["planning"]["difficulty_threshold"] = "Kitchen"

            for i in range(trials_per_env):
                trial_name = "search-%s_%d%d_%s" % (target_class, env_id, i, agent_type)


# def main():
#     parser = argparse.ArgumentParser(description="Run the object search with subgoals program.")
#     parser.add_argument("config_file",
#                         type=str, help="Path to .yaml configuration file (world distribution)")
#     parser.add_argument("diffc_score_file",
#                         type=str, help="Path to .csv for difficulty")
#     parser.add_argument("corr_score_file",
#                         type=str, help="Path to .csv for correlation")
#     parser.add_argument("subgoal_score_file",
#                         type=str, help="Path to .csv for subgoal selection")
#     parser.add_argument("-T", "--target-class", default="Salt",
#                         type=str, help="Target class to search for")
#     parser.add_argument("-N", "--num-trials", default=30,
#                         type=int, help="Number of environments to generate and test")
#     parser.add_argument("--output-dir", default="./results",
#                         type=str, help="Directory to output results")
#     args = parser.parse_args()

#     with open(args.config_file) as f:
#         config = yaml.load(f)

#     # Parameters
#     params = {
#         "max_depth": 16,
#         "nsteps": 100,
#         "discount_factor": 0.95,
#         "num_sims": 350,
#         "exploration_constant": 100
#     }

#     # target sensor and slam sensor params
#     target_sensor_config = {}
#     slam_sensor_config = {}
#     for sensor_name in config["sensors"]:
#         cfg = config["sensors"][sensor_name]
#         if args.target_class in cfg["noises"]:
#             target_sensor_config = copy.deepcopy(cfg)
#             target_sensor_config["noises"] = target_sensor_config["noises"][args.target_class]
#         if sensor_name.lower().startswith("room"):
#             slam_sensor_config = copy.deepcopy(cfg)

#     df_corr = pd.read_csv(args.corr_score_file)
#     df_dffc = pd.read_csv(args.diffc_score_file)
#     df_subgoal = pd.read_csv(args.subgoal_score_file)

#     reward_rows = []
#     success_rows = []
#     try:
#         for i in range(args.num_trials):
#             print("Generating environment that surely contains %s" % args.target_class)
#             env = generate_world(config, seed=100, required_classes={args.target_class})

#             # # MDP
#             # env_copy = copy.deepcopy(env)
#             # mdp_rewards = test_mdp(env_copy, args.target_class,
#             #                        target_sensor_config=target_sensor_config,
#             #                        slam_sensor_config=slam_sensor_config,
#             #                        **params)

#             # SUBGOAL (with update)
#             env_copy = copy.deepcopy(env)
#             subgoal_corr_rewards = test_subgoals_agent(env_copy, args.target_class, config,
#                                                        df_corr, df_dffc, df_subgoal,
#                                                        use_correlation_belief_update=True,
#                                                        **params)

#             # SUBGOAL (without correlation update)
#             env_copy = copy.deepcopy(env)
#             subgoal_nocorr_rewards = test_subgoals_agent(env_copy, args.target_class, config,
#                                                          df_corr, df_dffc, df_subgoal,
#                                                          use_correlation_belief_update=False,
#                                                          **params)

#             # POMDP
#             env_copy = copy.deepcopy(env)
#             pomdp_rewards = test_pomdp(env_copy, args.target_class,
#                                        target_sensor_config=target_sensor_config,
#                                        slam_sensor_config=slam_sensor_config,
#                                        **params)

#             # POMDP NK
#             env_copy = copy.deepcopy(env)
#             pomdp_nk_rewards = test_pomdp_nk(env_copy, args.target_class,
#                                              target_sensor_config=target_sensor_config,
#                                              slam_sensor_config=slam_sensor_config,
#                                              **params)


#             # mdp_disc = discounted_cumulative_reward(mdp_rewards, params["discount_factor"])
#             pomdp_disc = discounted_cumulative_reward(pomdp_rewards, params["discount_factor"])
#             pomdp_nk_disc = discounted_cumulative_reward(pomdp_nk_rewards, params["discount_factor"])
#             subgoal_nocorr_disc = discounted_cumulative_reward(subgoal_nocorr_rewards, params["discount_factor"])
#             subgoal_corr_disc = discounted_cumulative_reward(subgoal_corr_rewards, params["discount_factor"])
#             reward_rows.append([#mdp_disc,
#                 pomdp_disc, pomdp_nk_disc,
#                 subgoal_nocorr_disc, subgoal_corr_disc])

#             # mdp_success   = int(mdp_rewards[-1] == 100.0)
#             pomdp_success = int(pomdp_rewards[-1] == 100.0)
#             pomdp_nk_success = int(pomdp_nk_rewards[-1] == 100.0)
#             subgoal_nocorr_success = int(subgoal_nocorr_rewards[-1] == 100.0)
#             subgoal_corr_success = int(subgoal_corr_rewards[-1] == 100.0)
#             success_rows.append([#mdp_success,
#                 pomdp_success, pomdp_nk_success,
#                 subgoal_nocorr_success, subgoal_corr_success])
#     finally:
#         df_rewards = pd.DataFrame(reward_rows,
#                                   columns=["POMDP", "POMDP-NK", "SUBGOAL-NC", "SUBGOAL-C"])
#         df_success = pd.DataFrame(success_rows,
#                                   columns=["POMDP", "POMDP-NK", "SUBGOAL-NC", "SUBGOAL-C"])

#         start_time = dt.now()
#         timestr = start_time.strftime("%Y%m%d%H%M%S%f")[:-3]
#         df_rewards.to_csv(os.path.join(args.output_dir, "results_%s_disc-rewards.csv" % (timestr)))
#         df_success.to_csv(os.path.join(args.output_dir, "results_%s_success.csv" % (timestr)))


# if __name__ == "__main__":
#     main()
