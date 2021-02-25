"""
Learn spatial correlation by super pixels (i.e. bigger big cells)
The spatial correlation between two object types is proportional
to the number of times their instances appear in the same super pixel
(co-occurrence)
"""
from corrsearch.experiments.domains.thor.process_scenes import load_scene_info
from pprint import pprint


def cooccur_matrix(scenes, grid_size=0.25, scale=4):
    """
    Returns a cooccurrence matrix (mapping from [i,j] -> Score)
    for all pairs of object types in the scenes. This method
    is the same as Kollar and Roy (2009).
    """
    matrix = {}
    scene_infos = {scene_name : load_scene_info(scene_name)
                   for scene_name in scenes}
    for scene_name, scene_info in scene_infos.items():
        for type1 in scene_info.obj_types():
            for type2 in scene_info.obj_types():
                if (type2, type1) in matrix:
                    continue
                matrix[(type1, type2)] = 0
                for objid1 in scene_info.pomdp_objids(type1):
                    thor_pose1 = scene_info.thor_obj_pose2d(objid1)
                    grid_loc1 = (thor_pose1[0] // grid_size, thor_pose1[1] // grid_size)
                    supergrid_loc1 = grid_loc1[0] // scale, grid_loc1[1] // scale
                    for objid2 in scene_info.pomdp_objids(type2):
                        thor_pose2 = scene_info.thor_obj_pose2d(objid2)
                        grid_loc2 = (thor_pose2[0] // grid_size, thor_pose2[1] // grid_size)
                        supergrid_loc2 = grid_loc2[0] // scale, grid_loc2[1] // scale
                        if supergrid_loc1 == supergrid_loc2:
                            matrix[(type1, type2)] += 1
    return matrix

if __name__ == "__main__":
    scenes = []
    for i in range(1, 13):
        for j in range(1,6):
            scene = "FloorPlan_Train{}_{}".format(i, j)
            scenes.append(scene)
    matrix = cooccur_matrix(scenes)
    pprint(["{}:{}".format(k,matrix[k]) for k in sorted(matrix, key=matrix.get)])
