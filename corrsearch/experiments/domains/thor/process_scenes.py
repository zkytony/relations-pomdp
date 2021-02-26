"""
Process every scene. Assigns an integer to every object
in the scene.
"""
import os
import pickle
import yaml
import numpy as np
from corrsearch.objects.template import Object
from corrsearch.experiments.domains.thor.thor\
    import robothor_scene_names, ithor_scene_names, ThorSceneInfo

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

    # for i in range(1, 13):
    #     for j in range(1,6):
    #         scene = "FloorPlan_Train{}_{}".format(i, j)
    #         print(scene)
    #         mapping = process_scene(scene)
    # scenes = robothor_scene_names("Train")
    for scene in ithor_scene_names("kitchen", levels=np.arange(1,11)):
        mapping = ThorSceneInfo.extract_objects_info(scene)
        with open(os.path.join("data", "{}-objects.pkl".format(scene)), "wb") as f:
            pickle.dump(mapping, f)

if __name__ == "__main__":
    main()
