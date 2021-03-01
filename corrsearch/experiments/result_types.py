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
from corrsearch.utils import hex_to_rgb, ci_normal
from statannot import add_stat_annotation


method_to_name = {
    "heuristic#noprune#iq"                     : "Corr+Heuristic",
    "heuristic#noprune#iq-all-everywhere"      : "Corr+Heuristic+Everywhere",
    "corr-pouct"                               : "Corr",
    "target-only-pouct"                        : "Target",
    "target-only"                              : "Target",
    "target-only-heuristic-all-everywhere"     : "Target+Heuristic+Everywhere",
    "entropymin"                               : "Greedy",
    "random"                                   : "Random",
    "heuristic#k=2#iq"                         : "Corr+Heuristic(k=2)",
    "heuristic#k=2#noiq"                       : "Corr+Heuristic(k=2, NoIQ)",
}


name_to_color = {
    "Corr+Heuristic(k=2)":  np.array(hex_to_rgb("#8172B2")) / 255.,
    "Corr+Heuristic":      np.array(hex_to_rgb("#4C72B0")) / 255.,
    "Corr":                np.array(hex_to_rgb("#55A868")) / 255.,
    "Target":              np.array(hex_to_rgb("#C44E52")) / 255.,
    "Greedy":           np.array(hex_to_rgb("#CCB974")) / 255.,
    "Random":              np.array(hex_to_rgb("#909090")) / 255.
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
                rows.append([baseline, seed, cum_reward, success, fail])
        cls.sharedheader = ["baseline", "seed", "disc_reward", "success", "fail"]
        return rows

    @classmethod
    def save_gathered_results(cls, gathered_results, path):
        if "Field2D" in path:
            cls.save_gathered_results_field2d(gathered_results, path)
        elif "Thor" in path:
            cls.save_gathered_results_thor(gathered_results, path)

    ###### Thor
    @classmethod
    def save_gathered_results_thor(cls, gathered_results, path):
        all_rows = []
        for global_name in gathered_results:
            tokens = global_name.split("-")
            scene_type = tokens[0]
            target_class = tokens[1]
            scene_name = tokens[2]
            for row in gathered_results[global_name]:
                all_rows.append([scene_type, target_class, scene_name] + row)
        df = pd.DataFrame(all_rows,
                          columns=["scene_type", "target_class", "scene_name"] + cls.sharedheader)
        # Produce a summary table as follows:
        # target class scene_name baseline success% fail% reward SPL?/DistanceOfSuccess?
        summary = df.groupby(['target_class', 'baseline'])\
                    .agg([("avg", "mean"),
                          "std",
                          ("ci95", lambda x: ci_normal(x, confidence_interval=0.95))])
        summary = summary.unstack()

        for target_class in summary.index:
            fig, axes = plt.subplots(1,3, figsize=(15,7))
            disc_reward_avg = summary.loc[target_class].unstack().loc[("disc_reward", "avg")]
            disc_reward_ci95 = summary.loc[target_class].unstack().loc[("disc_reward", "ci95")]
            axes[0].axhline(y=0.0, color='k', linestyle='-')
            cls.plot_bar_stat(disc_reward_avg, disc_reward_ci95, axes[0], "Discounted Reward")

            success_avg = summary.loc[target_class].unstack().loc[("success", "avg")]
            cls.plot_bar_stat(success_avg, None, axes[1], "Success")
            axes[1].set_ylim(0, 1.1)
            axes[1].axhline(y=0.0, color='k', linestyle='-')

            fail_avg = summary.loc[target_class].unstack().loc[("fail", "avg")]
            cls.plot_bar_stat(fail_avg, None, axes[2], "Fail")
            axes[2].set_ylim(0, 1.1)
            axes[2].axhline(y=0.0, color='k', linestyle='-')
            plt.savefig(os.path.join(path, f"summary-{target_class}.png"))

        # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(summary)
        df.groupby(["target_class", "baseline"]).agg([("avg","mean"),
                                                      ("ci95", lambda x: ci_normal(x, confidence_interval=0.95))]).to_csv("numbers.csv", float_format='%.2f')

    @classmethod
    def plot_bar_stat(cls, avg_series, ci95_series, ax, title="title"):
        xvals = np.arange(len(avg_series))
        heights = []
        ticks = []
        yerr = None if ci95_series is None else []
        for baseline in avg_series.index:
            ticks.append(method_to_name[baseline])
            heights.append(avg_series[baseline])
            if ci95_series is not None:
                yerr.append(ci95_series[baseline])
        ax.bar(xvals, heights, tick_label=ticks, yerr=yerr)
        ax.set_title(title)


    ###### Field2D
    @classmethod
    def save_gathered_results_field2d(cls, gathered_results, path):
        all_rows = []
        prepend_header = []
        for global_name in gathered_results:
            prepend = []
            ylim = None
            if global_name.startswith("varysize"):
                size = global_name.split("-")[1]
                prepend.append(int(size.split(",")[0]))
                prepend_header = ["size"]
                xlabel = "Search Space Size"
                invert_x = False
                # Order the baselines
                baselines = ["Corr+Heuristic", "Corr",  "Target", "Greedy", "Random"]
                additional_x = None
                figsize=(20,8)
            elif global_name.startswith("varynobj"):
                nobj = global_name.split("-")[1]
                prepend.append(int(nobj))
                prepend_header = ["nobj"]
                xlabel = "Number of Objects"
                invert_x = False
                # Order the baselines
                baselines = [ #"Corr+Heuristic(k=2, NoIQ)",
                              "Corr+Heuristic(k=2)",
                              "Corr+Heuristic", "Corr",  "Target", "Greedy", "Random"]
                additional_x = None
                ylim = None#(-20, 60)
                figsize=(12,8)
            elif global_name.startswith("varynoise"):
                target_tp = float(global_name.split("-")[1][1:])
                other_tp = float(global_name.split("-")[2][1:])
                prepend.append(target_tp)  # tp = true positive
                prepend.append(other_tp)
                prepend_header = ["target_tp", "other_tp"]
                xlabel = "Target detector true positive rate"
                baselines = list(reversed(["Corr+Heuristic", "Corr",  "Target", "Greedy", "Random"]))
                invert_x = True
                additional_x = 1
                figsize=(12,8)

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


        summary = df.groupby([prepend_header[0], "baseline"]).agg([("avg", "mean"),
                                                                   ("ci95", lambda x: ci_normal(x, confidence_interval=0.95))])
        summary.to_csv("numbers.csv", float_format='%.2f')


        # plotting reward
        cls.plot_and_save(path, df, prepend_header[0], xlabel, invert_x, ylim=ylim, figsize=figsize)
        if additional_x is not None:
            cls.plot_and_save(path, df, prepend_header[additional_x], "Other object true positive rate", invert_x,
                              suffix="_otherobj", ylim=ylim, figsize=figsize)

            summary = df.groupby([prepend_header[additional_x], "baseline"]).agg([("avg", "mean"),
                                                                                  ("ci95", lambda x: ci_normal(x, confidence_interval=0.95))])
            summary.to_csv("numbers_otherobj.csv", float_format='%.2f')






    @classmethod
    def plot_and_save(cls, path, df, x, xlabel, invert_x, suffix="", ylim=None, figsize=(8,8)):

        cls._plot_summary(df, x=x, y="disc_reward",
                          title="Discounted Return",
                          xlabel=xlabel,
                          ylabel="Discounted Cumulative Reward",
                          filename=os.path.join(path, "rewards%s.png" % suffix),
                          figsize=figsize,
                          invert_x=invert_x,
                          add_stat_annot=True,
                          ylim=ylim)

        # success rate
        cls._plot_summary(df, x=x, y="success",
                          title="Success Rate",
                          xlabel=xlabel,
                          ylabel="Success Rate",
                          filename=os.path.join(path, "outcome_success%s.png" % suffix),
                          invert_x=invert_x,
                          figsize=figsize,
                          add_stat_annot=True,
                          ci=None,
                          ylim=(0,1.0))

        # failure rate
        cls._plot_summary(df, x=x, y="fail",
                          title="Incorrect Declaration Rate",
                          xlabel=xlabel,
                          ylabel="Incorrect Declaration Rate",
                          filename=os.path.join(path, "outcome_fail%s.png" % suffix),
                          figsize=figsize,
                          invert_x=invert_x,
                          add_stat_annot=True,
                          ci=None,
                          ylim=None)

    @classmethod
    def _plot_summary(cls, df, x, y, title, xlabel, ylabel, filename,
                      invert_x, ylim=None, add_stat_annot=True, plot_type="bar", figsize=(8,8), ci=95):
        sns.set(style="whitegrid")
        sns.set_context('paper', font_scale=2)
        fig, ax = plt.subplots(figsize=figsize)
        xvals = df[x].unique()
        ax.set_xticks(sorted(xvals))
        if plot_type == "point":
            g = sns.pointplot(x=x, y=y,
                              hue="baseline", ax=ax, data=df, ci=ci, capsize=.15,
                              palette=name_to_color)
        elif plot_type == "bar":
            g = sns.barplot(x=x, y=y,
                            hue="baseline", ax=ax, data=df, ci=ci,# capsize=.08,
                            saturation=0.8,
                            palette=name_to_color)
            # if add_stat_annot:
            #     boxpairs = []
            #     for xval in xvals:
            #         # pair1 = ((xval, "Corr+Heuristic"), (xval, "Corr"))
            #         pair2 = ((xval, "Corr+Heuristic"), (xval, "Corr+Heuristic(k=2)"))
            #         pair3 = ((xval, "Corr+Heuristic"), (xval, "Greedy"))
            #         boxpairs.extend([pair3,pair2])

            #     add_stat_annotation(ax, plot="barplot", data=df,
            #                         x=x, y=y, hue="baseline",
            #                         box_pairs=boxpairs,
            #                         loc="inside",
            #                         test="t-test_ind",
            #                         line_offset_to_box=0.05,
            #                         line_offset=0.02,
            #                         offset_basis="ymean",
            #                         verbose=2)
        if x == "size":
            ax.set_xticklabels(["{}x{}".format(x, x) for x in sorted(xvals)])
        l = ax.legend()
        l.set_title("")
        # Put a legend to the right of the current axis
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        # l.remove()
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        if invert_x:
            ax.invert_xaxis()
        plt.tight_layout()
        plt.savefig(filename)


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
