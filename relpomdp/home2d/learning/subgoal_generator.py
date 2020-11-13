# Contains a function to generate subgoal,
# based on correlation and difficulty scores
from relpomdp.home2d.planning.test_utils import difficulty, correlation, remap
import pandas as pd
import yaml
import os
import argparse
from datetime import datetime as dt

def correlation_score(target_class, other_class, df_corr, minval=0., maxval=1.):
    corr = correlation(df_corr, target_class, other_class)
    corr_min = df_corr["corr_score"].min()
    corr_max = df_corr["corr_score"].max()
    corr = remap(corr, corr_min, corr_max, minval, maxval)
    return corr

def difficulty_score(target_class, df_difficulty, minval=0., maxval=1.):
    dffc = difficulty(df_difficulty, target_class)
    dffc_min = df_difficulty["difficulty"].min()
    dffc_max = df_difficulty["difficulty"].max()
    dffc = remap(dffc, dffc_min, dffc_max, minval, maxval)
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
    difficulty = difficulty_score(other_class, df_difficulty)
    return corr_weight*correlation - difficulty_weight*difficulty

# Test
def main():
    parser = argparse.ArgumentParser(description="Compute Subgoal Scores")
    parser.add_argument("config_file",
                        type=str, help="Path to .yaml configuration file")
    parser.add_argument("corr_df_path",
                        type=str, help="Path to correlation csv file")
    parser.add_argument("dffc_df_path",
                        type=str, help="Path to difficulty csv file")
    parser.add_argument("output_dir",
                        type=str, help="Directory to output computed subgoal scores saved in a file")
    args = parser.parse_args()

    df_corr = pd.read_csv(args.corr_df_path)
    df_dffc = pd.read_csv(args.dffc_df_path)

    with open(args.config_file) as f:
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
    start_time = dt.now()
    timestr = start_time.strftime("%Y%m%d%H%M%S%f")[:-3]
    config_filename = "C#%s" % (os.path.splitext(os.path.basename(args.config_file))[0])
    difficulty_filename = "D#%s" % "-".join(os.path.splitext(os.path.basename(args.dffc_df_path))[0].split("-")[1:5])
    df.to_csv(os.path.join(args.output_dir, "./scores_%s_%s_%s.csv" % (config_filename, difficulty_filename, timestr)))

if __name__ == "__main__":
    main()
