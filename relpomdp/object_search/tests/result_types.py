from sciex import *
from sciex.util import *
import numpy as np
from scipy import stats
import math
import os
import json
import yaml
import copy
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)


class HistoryResult(PklResult):

    DISCOUNT_FACTOR = 0.95
    
    def __init__(self, history):
        """list of state objects"""
        super().__init__(history)

    @classmethod
    def FILENAME(cls):
        return "history.pkl"

    @classmethod
    def gather(cls, results):
        """`results` is a mapping from specific_name to a dictionary {seed: actual_result}.
        Returns a more understandable interpretation of these results"""
        # Returns the number of objects detected at the end.
        myresults = {}
        for specific_name in results:
            all_rewards = []
            found_indices = []
            seeds = []
            for seed in results[specific_name]:
                seeds.append(seed)
                history = results[specific_name][seed]
                disc_reward = 0
                discount = 1.0
                step = 0
                for state, action, observation, reward in history:
                    disc_reward += discount*reward
                    discount *= HistoryResult.DISCOUNT_FACTOR
                    if step == len(history)-1:
                        if reward == 100:
                            found_indices.append(step)
                        else:
                            found_indices.append(-1)
                    step += 1
                all_rewards.append(disc_reward)
            myresults[specific_name] = {"seeds": seeds,
                                        "all_rewards": all_rewards,
                                        "found_indices": found_indices}
        return myresults
    
    @classmethod
    def save_gathered_results(cls, gathered_results, path):
        df = cls.organize(gathered_results)
        df.to_csv(os.path.join(path, "history_results.csv"))
        
        df_rewards = cls.summarize_disc_reward(df)
        df_rewards.to_csv(os.path.join(path, "rewards-discounted.csv"))

    @classmethod
    def organize(cls, gathered_results):
        baselines = set()
        casemap = {}  # map from case to prior_types; Used to make sure we only
                      # use trials where all baselines have results        
        for global_name in gathered_results:
            init_robot_pose = global_name.split("-")[2]
            for specific_name in gathered_results[global_name]:
                baseline = specific_name
                results = gathered_results[global_name][specific_name]
                seeds = results['seeds']
                all_rewards = results['all_rewards']
                found_indices = results['found_indices']
                baselines.add(baseline)

                for i, seed in enumerate(seeds):
                    # founds is a list of (num_found, step) tuples; We only
                    # care about the step number, because the num_found increases by 1 always.
                    disc_reward = all_rewards[i]
                    found_idx = found_indices[i]
                    case = (init_robot_pose, seed)
                    if case not in casemap:
                        casemap[case] = []
                    if found_idx == -1:
                        num_detected = 0
                    else:
                        num_detected = 1  # TODO: ONLY FOR SINGLE OBJECT SEARCH
                    casemap[case].append((baseline, disc_reward, found_idx, num_detected))

        # Make sure we are comparing between cases where all prior types have result.
        rows = []
        counts = {}  # maps from (sensor_range, map_name, prior_type) -> number
        for case in casemap:
            if set(t[0] for t in casemap[case]) != baselines:
                # We do not have all prior types for this case.
                continue
            for baseline, disc_reward, found_idx, num_detected in casemap[case]:
                init_robot_pose, seed = case
                rows.append([init_robot_pose, baseline, seed, disc_reward, found_idx, num_detected])
        df = pd.DataFrame(rows, columns=["init_robot_pose",
                                         "baseline",
                                         "seed",
                                         "disc_reward",
                                         "found_index",
                                         "num_detected"])
        return df
        
    @classmethod
    def summarize_disc_reward(cls, df):
        df = df.copy()
        df = df.drop(columns=["init_robot_pose", "found_index", "num_detected"])
        summary = df.groupby(["baseline"])\
                    .agg([("ci95", lambda x: ci_normal(x, confidence_interval=0.95)),
                          ("ci90", lambda x: ci_normal(x, confidence_interval=0.9)),
                          ("sem", lambda x: stderr(x)),
                          ('avg', 'mean'),
                          ('median', lambda x: np.median(x)),
                          'std',
                          'count',
                          'sum'])  # this is not relevant for most except for foref_prediction_count
        flatten_column_names(summary)        
        return summary

    # @classmethod
    # def plot_reward(cls, df):
    #     df = cls.summarize
