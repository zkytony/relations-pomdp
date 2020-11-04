# Contains a function to generate subgoal,
# based on correlation and difficulty scores
from relpomdp.home2d.learning.testing.test_utils import difficulty, correlation, remap
import pandas as pd
import yaml
import os

def correlation_score(target_class, other_class, df_corr, minval=0., maxval=1.):
    corr = correlation(df_corr, target_class, other_class)
    corr_min = df_corr["corr_score"].min()
    corr_max = df_corr["corr_score"].max()
    corr = remap(corr, corr_min, corr_max, minval, maxval)
    return corr

def difficulty_score(target_class, other_class, df_difficulty, minval=0., maxval=1.):
    dffc = difficulty(df_corr, target_class, other_class)
    dffc_min = df_corr["difficulty"].min()
    dffc_max = df_corr["difficulty"].max()
    dffc = remap(dffc, corr_min, corr_max, minval, maxval)
    return dffc

def score(target_class, other_class,
          df_corr, df_difficulty,
          corr_weight=0.5, difficulty_weight=0.5):
    """
    Returns a score that tells you how good it is to search for
    the other class first before searching for the target class,
    based on the table of correlation and difficult
    """
    correlation = correlation_score(target_class, other_class, df_corr)
    difficulty = difficulty_score(target_class, other_class, df_difficulty)
    return corr_weight*correlation - difficulty_weight*difficulty

# Test
def test():
    corr_df_path = "../data/correlation-try1-10-20-2020.csv"
    dffc_df_path = "../data/difficulty-try1-10-20-2020-20201026162744897.csv"

    df_corr = pd.read_csv(corr_df_path)
    df_dffc = pd.read_csv(dffc_df_path)

    config_path = "../configs/10x10_10-20-2020.yaml"
    with open(config_path) as f:
        config = yaml.load(f)

    objects = set()
    for room_type in config["objects"]:
        for objclass in config["objects"][room_type]:
            objects.add(objclass)
        objects.add(room_type)
    print(objects)

    rows = []
    for c1 in objects:
        for c2 in objects:
            rows.append((c1, c2, score(c1, c2, df_corr, df_dffc)))
    df = pd.DataFrame(rows, columns=["c1", "c2", "score"])
    df.to_csv("./scores.csv")

if __name__ == "__main__":
    test()
