"""
Process every scene. Assigns an integer to every object
in the scene.
"""
import os
import pickle
from corrsearch.experiments.domains.thor.thor import *

class SceneInfo:
    def __init__(self, scene_name, type_obj_map):
        self.scene_name = scene_name
        self.type_obj_map = type_obj_map

        # Map from objid (pomdp) to thor object dict
        self._idp2t = {}
        for objtype in self.type_obj_map:
            for objid in self.type_obj_map[objtype]:
                assert objid not in self._idp2t
                self._idp2t[objid] = self.type_obj_map[objtype][objid]

    def pomdp_objids(self, objtype):
        """Returns objids (pomdp) for the given object type"""
        return set(self.type_obj_map[objtype].keys())

    def objid_for_type(self, objtype):
        """Returns a pomdp object id for the given type;
        If there are multiple ones, return the smallest."""
        return min(self.pomdp_objids(objtype))

    def obj(self, objid):
        """Returns the THOR object data structure given objid (pomdp)"""
        return self._idp2t[objid]

    def obj_type(self, objid):
        return self._idp2t[objid]["objectType"]


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

    controller.stop()
    return type_obj_map


def load_scene_info(scene_name, data_path="data"):
    """Returns scene info, which is a mapping:
    A mapping {object_type -> {objid (pomdp) -> obj_dict}}"""
    with open(os.path.join(
            data_path, "{}-objects.pkl".format(scene_name)), "rb") as f:
        type_obj_map = pickle.load(f)
    return SceneInfo(scene_name, type_obj_map)


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