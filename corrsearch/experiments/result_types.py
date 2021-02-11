from sciex import *
import numpy as np
from scipy import stats
import math
import os
import json
import yaml
import copy
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


#### Actual results for experiments ####
class RewardsResult(YamlResult):
    def __init__(self, rewards):
        """rewards: a list of reward floats"""
        self._rewards = rewards
        super().__init__(rewards)

    @classmethod
    def FILENAME(cls):
        return "rewards.yaml"

    @classmethod
    def collect(cls, path):
        # This should be standardized.
        with open(path) as f:
            rewards = yaml.load(f)
        trial_path = os.path.dirname(path)
        with open(os.path.join(trial_path, "trial.pkl"), "rb") as f:
            config = pickle.load(f).config
        return (rewards, trial_path, config)

    @classmethod
    def gather(cls, results):
        rows = []
        for baseline in results:
            for seed in results[baseline]:
                rewards, trial_path, config = results[baseline][seed]
                discount_factor = config["discount_factor"]
                # compute cumulative reward discounted
                cum_reward = 0.0
                discount = 1.0
                for r in rewards:
                    cum_reward += discount*r
                    discount *= discount_factor
                rows.append([baseline, int(seed), cum_reward])
        return rows

    @classmethod
    def save_gathered_results(cls, gathered_results, path):
        all_rows = []
        prepend_header = []
        for global_name in gathered_results:
            prepend = []
            if global_name.startswith("varynoise"):
                noise = float(global_name.split("-")[1])
                prepend.append(noise)
                prepend_header = ["noise"]
                xlabel = "Target detector True Positive Rate"
                invert_x = True
            elif global_name.startswith("varysize"):
                size = global_name.split("-")[1]
                prepend.append(int(size.split(",")[0]))
                prepend_header = ["size"]
                xlabel = "Size"
                invert_x = False

            for row in gathered_results[global_name]:
                all_rows.append(prepend + row)
        df = pd.DataFrame(all_rows,
                          columns=prepend_header + ["baseline", "seed", "disc_reward"])
        df.to_csv(os.path.join(path, "rewards.csv"))
        # plotting
        fig, ax = plt.subplots(figsize=(5.5,4))
        sns.barplot(x=prepend_header[0], y="disc_reward",
                    hue="baseline", ax=ax, data=df, ci=95)
        ax.set_ylabel("Discounted Cumulative Reward")
        ax.set_xlabel(xlabel)
        if invert_x:
            ax.invert_xaxis()
        plt.savefig(os.path.join(path, "rewards.png"))


class StatesResult(PklResult):
    def __init__(self, states):
        """list of state objects"""
        super().__init__(states)

    @classmethod
    def FILENAME(cls):
        return "states.pkl"

    @classmethod
    def gather(cls, results):
        pass

    @classmethod
    def save_gathered_results(cls, gathered_results, path):
        pass


class HistoryResult(PklResult):
    def __init__(self, history):
        """list of state objects"""
        super().__init__(history)

    @classmethod
    def FILENAME(cls):
        return "history.pkl"
