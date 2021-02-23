"""
Process every scene. Assigns an integer to every object
in the scene.
"""
import os
import pickle
from corrsearch.experiments.domains.thor.thor import *

def process_scene(scene):
    """Returns a mapping from object type to {objid -> objinfo}
    where `objid` is an integer id and `objinfo` is the metadata
    of the object obtained at the initial state of the scene"""
    controller = launch_controller({"scene_name": scene})
    event = controller.step(action="Pass")

    # Maps from object type to an integer (e.g. 1000) that
    # is the the id of the FIRST INSTANCE of that type
    type_id_map = {}
    # Maps from object type to objects
    type_obj_map = {}
    for obj in event.metadata["objects"]:
        objtype = obj["objectType"]
        if objtype not in type_id_map:
            type_id_map[objtype] = (len(type_id_map)+1)*1000
            type_obj_map[objtype] = {}

        objid = len(type_obj_map[objtype]) + type_id_map[objtype]
        type_obj_map[objtype][objid] = obj

    return type_obj_map


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
