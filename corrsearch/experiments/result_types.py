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
from corrsearch.utils import hex_to_rgb
from statannot import add_stat_annotation


method_to_name = {
    "heuristic#noprune#iq"  : "Corr+Heuristic",
    "corr-pouct"            : "Corr",
    "target-only-pouct"     : "Target",
    "entropymin"            : "Greedy",
    "random"                : "Random",
    "heuristic#k=2#iq"      : "Corr+Heuristic(k=2)",
    "heuristic#k=2#noiq"    : "Corr+Heuristic(k=2, NoIQ)",
}

name_to_color = {
    "Corr+Heuristic(k=2, NoIQ)" :  np.array(hex_to_rgb("#77a16c")) / 255.,
    "Corr+Heuristic(k=2)"       :  np.array(hex_to_rgb("#8bbd28")) / 255.,
    "Corr+Heuristic"            :  np.array(hex_to_rgb("#21db65")) / 255.,
    "Corr"                      :  np.array(hex_to_rgb("#80ad91")) / 255.,
    "Target"                    :  np.array(hex_to_rgb("#d9cc6c")) / 255.,
    "Greedy"                    :  np.array(hex_to_rgb("#66a7e8")) / 255.,
    "Random"                    :  np.array(hex_to_rgb("#d15f4b")) / 255.,
}

# Order the baselines
baselines = ["Random", "Greedy", "Target", "Corr",
             "Corr+Heuristic", "Corr+Heuristic(k=2)", "Corr+Heuristic(k=2, NoIQ)"]


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

                success = 0  # Found, correct
                fail = 0     # Found, incorrect

                for r in rewards:
                    cum_reward += discount*r
                    discount *= discount_factor
                    if r == 100.0:
                        success = 1  # means found and correct
                    elif r == -100.0:
                        fail = 1  # means found, but wrong
                rows.append([baseline, int(seed), cum_reward, success, fail])
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
                # Order the baselines
                baselines = ["Random", "Greedy", "Target", "Corr",
                             "Corr+Heuristic"]# "Corr+Heuristic(k=2)", "Corr+Heuristic(k=2, NoIQ)"]
            elif global_name.startswith("varysize"):
                size = global_name.split("-")[1]
                prepend.append(int(size.split(",")[0]))
                prepend_header = ["size"]
                xlabel = "Size"
                invert_x = False
                # Order the baselines
                baselines = ["Random", "Greedy", "Target", "Corr",
                             "Corr+Heuristic"]

            for row in gathered_results[global_name]:
                all_rows.append(prepend + row)

        df = pd.DataFrame(all_rows,
                          columns=prepend_header + ["baseline", "seed", "disc_reward", "success", "fail"])

        df["baseline"] = df["baseline"].replace(method_to_name)
        df = df.loc[df["baseline"].isin(set(baselines))]
        # Sort data frame based on the order given in baselines
        df = df.sort_values(by="baseline",
                            key=lambda col: pd.Series(baselines.index(col[i])
                                                      for i in range(len(col))))
        df.to_csv(os.path.join(path, "rewards.csv"))

        # plotting reward
        cls._plot_summary(df, x=prepend_header[0], y="disc_reward",
                          title="Discounted Return",
                          xlabel=xlabel,
                          ylabel="Discounted Cumulative Reward",
                          filename=os.path.join(path, "rewards.png"),
                          invert_x=invert_x,
                          add_stat_annot=True)

        # success rate
        cls._plot_summary(df, x=prepend_header[0], y="success",
                          title="Success Rate",
                          xlabel=xlabel,
                          ylabel="Success Rate",
                          filename=os.path.join(path, "outcome_success.png"),
                          invert_x=invert_x,
                          add_stat_annot=True)

        # failure rate
        cls._plot_summary(df, x=prepend_header[0], y="fail",
                          title="Incorrect Declaration Rate",
                          xlabel=xlabel,
                          ylabel="Incorrect Declaration Rate",
                          filename=os.path.join(path, "outcome_fail.png"),
                          invert_x=invert_x,
                          add_stat_annot=True)

    @classmethod
    def _plot_summary(cls, df, x, y, title, xlabel, ylabel, filename,
                      invert_x, add_stat_annot=True):
        fig, ax = plt.subplots(figsize=(7,5))
        sns.barplot(x=x, y=y,
                    hue="baseline", ax=ax, data=df, ci=95,
                    palette=name_to_color,
                    alpha=0.7)

        xvals = df[x].unique()
        if add_stat_annot:
            boxpairs = []
            for xval in xvals:
                pair1 = ((xval, "heuristic#noprune#iq"), (xval, "corr-pouct"))
                pair2 = ((xval, "heuristic#noprune#iq"), (xval, "entropymin"))
                pair3 = ((xval, "corr-pouct"), (xval, "target-only-pouct"))
                boxpairs.extend([pair1,pair2,pair3])

            add_stat_annotation(ax, plot="barplot", data=df,
                                x=x, y=y, hue="baseline",
                                box_pairs=boxpairs,
                                loc="inside",
                                test="t-test_ind",
                                line_offset_to_box=0.05,
                                line_offset=0.02,
                                offset_basis="ymean",
                                verbose=2)

        ax.set_ylabel(title)
        ax.set_xlabel(xlabel)
        if invert_x:
            ax.invert_xaxis()
        plt.tight_layout()
        plt.grid()
        plt.savefig(filename)

        #os.path.join(path, "rewards.png"))




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
