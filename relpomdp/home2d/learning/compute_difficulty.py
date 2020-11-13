import argparse
import pickle
import numpy as np
import pandas as pd
from compute_correlations import get_classes
from datetime import datetime as dt
import os

def process_detections_random(detections):
    """Detections is created by running a random explorer"""
    # For each class of object, find the step number that it is
    # first detected in the trial.
    class_detections = {}  # maps from class name to a list of steps that marks its detection
    for envid in detections:
        done_classes = set()
        for step in range(len(detections[envid])):
            d = detections[envid][step]
            _, detected_ids, detected_poses = d
            for detected_class in detected_poses:
                if detected_class in done_classes:
                    continue
                if detected_class not in class_detections:
                    class_detections[detected_class] = []
                class_detections[detected_class].append(step)
                done_classes.add(detected_class)
    return class_detections

def process_detections_pomdp(detections):
    """Detections is created by running a pomdp solver"""
    class_detections = {}
    for envid in detections:
        done_classes = set()
        for detected_class, objid, step in detections[envid]:
            if detected_class in done_classes:
                continue
            if detected_class not in class_detections:
                class_detections[detected_class] = []
            class_detections[detected_class].append(step)
            done_classes.add(detected_class)
    return class_detections

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

    detection_source = "none"
    if "random" in args.path_to_detections:
        class_detections = process_detections_random(detections)
        detection_source = "random"
    elif "pomdp" in args.path_to_detections:
        class_detections = process_detections_pomdp(detections)
        detection_source = "pomdp"

    # Simply compute the average step as the difficulty
    rows = []
    for objclass in class_detections:
        rows.append((objclass, np.mean(class_detections[objclass])))
    df = pd.DataFrame(rows, columns=["class", "difficulty"])
    df.to_csv(os.path.join(args.output_dir, "difficulty-%s-%s-%s.csv" % (filename, detection_source, timestr)))

if __name__ == "__main__":
    main()
