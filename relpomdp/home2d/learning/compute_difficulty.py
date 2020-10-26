import argparse
import pickle
import numpy as np
import pandas as pd
from compute_correlations import get_classes
from datetime import datetime as dt
import os

def main():
    parser = argparse.ArgumentParser(description="Compute difficulty of finding objects based on detection results")
    parser.add_argument("path_to_detections",
                        type=str, help="Path to .pkl file that stores detections")
    parser.add_argument("path_to_envs",
                        type=str, help="Path to a .pickle file that contains a collection of environments")
    parser.add_argument("output_dir",
                        type=str, help="Directory to output computed difficulty saved in a file")

    args = parser.parse_args()

    start_time = dt.now()
    timestr = start_time.strftime("%Y%m%d%H%M%S%f")[:-3]

    with open(args.path_to_envs, "rb") as f:
        envs = pickle.load(f)
    with open(args.path_to_detections, "rb") as f:
        detections = pickle.load(f)

    filename = os.path.splitext(os.path.basename(args.path_to_envs))[0]

    classes, _ = get_classes(envs)

    # For each class of object, find the step number that it is
    # first detected in the trial.
    class_detections = {}  # maps from class name to a list of steps that marks its detection
    for envid in detections:
        done_classes = set()
        for step in range(len(detections[envid])):
            d = detections[envid][step]
            for objid in d:
                objo = d[objid]
                if objo.objclass in done_classes:
                    continue
                if objo.objclass not in class_detections:
                    class_detections[objo.objclass] = []
                class_detections[objo.objclass].append(step)
                done_classes.add(objo.objclass)

    # Simply compute the average step as the difficulty
    rows = []
    for objclass in class_detections:
        rows.append((objclass, np.mean(class_detections[objclass])))
    df = pd.DataFrame(rows, columns=["class", "difficulty"])
    df.to_csv(os.path.join(args.output_dir, "difficulty-%s-%s.csv" % (filename, timestr)))

if __name__ == "__main__":
    main()
