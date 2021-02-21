"""
This script gathers statistics of robothor environments
"""
import os
import cv2
from ai2thor.controller import Controller
import pandas as pd

def gather_object_info(scene):
    controller = Controller(
        agentMode="bot",
        scene=scene
    )
    event = controller.step(action="Pass")

    rows = []
    for obj in event.metadata["objects"]:
        bbox = obj["axisAlignedBoundingBox"]
        rows.append([scene,
                     obj["objectId"],
                     obj["objectType"],
                     obj["position"]["x"],
                     obj["position"]["y"],
                     obj["position"]["z"],
                     bbox["center"]["x"],
                     bbox["center"]["y"],
                     bbox["center"]["z"],
                     bbox["size"]["x"],
                     bbox["size"]["y"],
                     bbox["size"]["z"]])
    controller.stop()
    return rows

def process_all():

    header = ["scene",
              "id",
              "type",
              "pos_x", "pos_y", "pos_z",
              "ctr_x", "ctr_y", "ctr_z",
              "size_x", "size_y", "size_z"]
    all_rows = []
    # training
    for i in range(1, 13):
        for j in range(1,6):
            scene = "FloorPlan_Train{}_{}".format(i, j)
            print(scene)
            try:
                rows = gather_object_info(scene)
                all_rows.extend(rows)
            except Exception as ex:
                print("Error for scene {}: {}".format(scene, ex))
    df = pd.DataFrame(all_rows, columns=header)
    df.to_csv("robothor_object_stats-2.csv", sep=",")


if __name__ == "__main__":
    process_all()
