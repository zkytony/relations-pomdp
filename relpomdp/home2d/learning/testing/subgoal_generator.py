# Contains a function to generate subgoal,
# based on correlation and difficulty scores
import pandas as pd
import yaml
import os

def remap(oldval, oldmin, oldmax, newmin, newmax):
    return (((oldval - oldmin) * (newmax - newmin)) / (oldmax - oldmin)) + newmin

def score(target_class, other_class,
          df_corr, df_difficulty,
          corr_weight=0.5, difficulty_weight=0.5):
    """
    Returns a score that tells you how good it is to search for
    the other class first before searching for the target class,
    based on the table of correlation and difficult
    """
    try:
        correlation = float(df_corr.loc[(df_corr["class1"] == target_class)\
                                        & (df_corr["class2"] == other_class)]["corr_score"])
        difficulty = float(df_difficulty.loc[df_difficulty["class"] == other_class]["difficulty"])
    except Exception:
        correlation = 0
        difficulty = 1000

    # remap the values
    corr_min = df_corr["corr_score"].min()
    corr_max = df_corr["corr_score"].max()
    dffc_min = df_difficulty["difficulty"].min()
    dffc_max = df_difficulty["difficulty"].max()

    correlation = remap(correlation, corr_min, corr_max, 0, 1.)
    difficulty = remap(difficulty, dffc_min, dffc_max, 0, 1.)

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
