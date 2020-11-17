from sciex import Experiment, Trial, Event, Result,\
    YamlResult, PklResult, PostProcessingResult
from relpomdp.utils import ci_normal
from relpomdp.home2d.experiments.pd_utils import flatten_column_names
from relpomdp.home2d.experiments.constants import METHOD_TO_NAME
from statannot import add_stat_annotation
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt

class RewardsResult(YamlResult):
    def __init__(self, rewards):
        """rewards: a list of reward floats"""
        super().__init__(rewards)

    @classmethod
    def FILENAME(cls):
        return "rewards.yaml"

    @classmethod
    def discounted_reward(cls, rewards, gamma=0.95):
        discount = 1.0
        cum_disc = 0.0
        for reward in rewards:
            cum_disc += discount * reward
            discount *= gamma
        return cum_disc

    @classmethod
    def gather(cls, results):
        """`results` is a mapping from specific_name to a dictionary {seed: actual_result}.
        Returns a more understandable interpretation of these results"""
        rows = []
        for specific_name in results:
            agent_type = specific_name
            for seed in results[specific_name]:
                rewards = results[specific_name][seed]
                discount_factor = 0.95
                disc_reward = RewardsResult.discounted_reward(rewards, gamma=discount_factor)
                rows.append((agent_type, seed, disc_reward))
        return rows

    @classmethod
    def save_gathered_results(cls, gathered_results, path):
        all_rows = []
        for global_name in gathered_results:
            # Oops - TODO: You should fix this when generating trials
            global_name_parsing = global_name.replace("Single-bed", "Single#bed")
            global_name_parsing = global_name_parsing.replace("Countertop-wood", "Countertop#wood")

            target_class = global_name_parsing.split("-")[1]
            world_width = global_name_parsing.split("-")[2][1:]
            world_length = global_name_parsing.split("-")[3][1:]

            case_rows = gathered_results[global_name]
            for row in case_rows:
                all_rows.append(row + (target_class, world_width, world_length))
        df = pd.DataFrame(all_rows,
                          columns=["agent_type", "seed", "disc_reward",
                                   "target_class", "world_width", "world_length"])
        df.to_csv(os.path.join(path, "rewards.csv"))
        grouped = df.groupby(["agent_type", "target_class", "world_width", "world_length"])
        agg = grouped.agg([("ci95", lambda x: ci_normal(x, confidence_interval=0.95)),
                           ("ci90", lambda x: ci_normal(x, confidence_interval=0.90)),
                           ('avg', 'mean')])
        flatten_column_names(agg)
        agg.to_csv(os.path.join(path, "rewards-summary.csv"))

        baselines = ["Rand", "Heur", "NS", "S", "S+B"]
        df["agent_type"] = df["agent_type"].replace(METHOD_TO_NAME)
        df = df.sort_values(by="agent_type",
                            key=lambda col: pd.Series(baselines.index(col[i])
                                                      for i in range(len(col))))
        cls._plot_summary(df,
                          "Overall Performance (%s x %s)" % (world_width, world_length),
                          "rewards", path,
                          add_stat_annot=(grouped.size()[-1] >= 15))

        # Easier sensor (single bed or computer)
        df_easy_sensor = df.loc[df["target_class"].isin({"Single-bed", "Computer"})]
        cls._plot_summary(df_easy_sensor,
                          "Single-bed & Computer (better detector) (%s x %s)"
                          % (world_width, world_length),
                          "rewards-better-sensor", path,
                          add_stat_annot=(grouped.size()[-1] >= 15))

        # Harder sensor (single bed or computer)
        df_hard_sensor = df.loc[df["target_class"].isin({"Salt", "Pepper"})]
        cls._plot_summary(df_hard_sensor,
                          "Salt & Pepper (worse detector) (%s x %s)" % (world_width, world_length),
                          "rewards-worse-sensor", path,
                          add_stat_annot=(grouped.size()[-1] >= 15))

        # Fewer subgoals (single bed or salt)
        df_few_subgoals = df.loc[df["target_class"].isin({"Salt", "Single-bed"})]
        cls._plot_summary(df_few_subgoals,
                          "Salt & Single-bed (nsubgoals=1) (%s x %s)" % (world_width, world_length),
                          "rewards-few-subgoals", path,
                          add_stat_annot=(grouped.size()[-1] >= 15))

        # More subgoals (Pepper or computer)
        df_more_subgoals = df.loc[df["target_class"].isin({"Pepper", "Computer"})]
        cls._plot_summary(df_more_subgoals,
                          "Pepper & Computer (nsubgoals=2) (%s x %s)" % (world_width, world_length),
                          "rewards-more-subgoals", path,
                          add_stat_annot=(grouped.size()[-1] >= 15))

    @classmethod
    def _plot_summary(cls, df, title, filename, savepath, add_stat_annot=False):
        """
        baselines: ["Rand", "Heur", "NS", "S", "S+B"]
        """
        # Plotting
        fig, ax = plt.subplots(figsize=(5.5,4))
        sns.barplot(x="agent_type", y="disc_reward", ci=95,
                    data=df, ax=ax)
        ## Add statistical significance annotation, when there's enough trials
        if add_stat_annot:
            boxpairs = [
                ("S+B", "Heur"),
                # ("S+B", "S"),
                # ("S+B", "NS")
            ]
            add_stat_annotation(ax, plot="barplot", data=df,
                                x="agent_type", y="disc_reward",
                                box_pairs=boxpairs,
                                loc="inside",
                                test="t-test_ind",
                                line_offset_to_box=0.05,
                                line_offset=0.02,
                                offset_basis="ymean",
                                verbose=2)
        ax.set_title(title)
        plt.savefig(os.path.join(savepath, "%s.png" % filename))
