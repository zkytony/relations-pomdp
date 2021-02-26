"""
Process every scene. Assigns an integer to every object
in the scene.
"""
import os
import pickle
import yaml
from corrsearch.objects.template import Object
# from corrsearch.experiments.

def shared_objects_in_scenes(scenes):
    objects = None
    for scene in scenes:
        scene_info = load_scene_info(scene)
        if objects is None:
            objects = scene_info.obj_types()
        else:
            objects = objects.intersection(scene_info.obj_types())
    return objects

def main():
    os.makedirs("data", exist_ok=True)
    for i in range(1, 13):
        for j in range(1,6):
            scene = "FloorPlan_Train{}_{}".format(i, j)
            print(scene)
            mapping = process_scene(scene)
            with open(os.path.join("data", "{}-objects.pkl".format(scene)), "wb") as f:
                pickle.dump(mapping, f)

if __name__ == "__main__":
    main()
