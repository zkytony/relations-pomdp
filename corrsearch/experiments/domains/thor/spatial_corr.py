"""
Learn spatial correlation by super pixels (i.e. bigger big cells)
The spatial correlation between two object types is proportional
to the number of times their instances appear in the same super pixel
(co-occurrence)

Also, involves functions that build the joint distribution (factored graph)
"""
from corrsearch.experiments.domains.thor.process_scenes import load_scene_info
from corrsearch.experiments.domains.thor.thor import robothor_scene_names
from corrsearch.models import *
from corrsearch.probability import *
from corrsearch.utils import indicator
from pprint import pprint

def cooccur(grid_loc1, grid_loc2, scale=4):
    """Given two grid locations, return True if they fall in the same
    bigger grid that is `scale` times bigger."""
    supergrid_loc1 = grid_loc1[0] // scale, grid_loc1[1] // scale
    supergrid_loc2 = grid_loc2[0] // scale, grid_loc2[1] // scale
    return supergrid_loc1 == supergrid_loc2

def cooccur_matrix(scenes, grid_size=0.25, scale=4):
    """
    Returns a cooccurrence matrix (mapping from [i,j] -> (#cooccur, #not-cooccur))
    for all pairs of object types in the scenes.

    This actually differs from Kollar and Roy (2009); We explicitly count the
    number of times two objects are NOT co-occurring too. That makes the factors
    simpler. TBH the factors defined in Kollar and Roy (2009) are not understandable to me.
    """
    matrix = {}
    scene_infos = {scene_name : load_scene_info(scene_name)
                   for scene_name in scenes}
    for scene_name, scene_info in scene_infos.items():
        for type1 in scene_info.obj_types():
            for type2 in scene_info.obj_types():
                if (type2, type1) in matrix:
                    continue
                matrix[(type1, type2)] = [0,0]
                for objid1 in scene_info.pomdp_objids(type1):
                    thor_pose1 = scene_info.thor_obj_pose2d(objid1)
                    grid_loc1 = (thor_pose1[0] // grid_size, thor_pose1[1] // grid_size)
                    for objid2 in scene_info.pomdp_objids(type2):
                        if objid1 == objid2:
                            continue
                        thor_pose2 = scene_info.thor_obj_pose2d(objid2)
                        grid_loc2 = (thor_pose2[0] // grid_size, thor_pose2[1] // grid_size)
                        if cooccur(grid_loc1, grid_loc2, scale=scale):
                            matrix[(type1, type2)][0] += 1
                        else:
                            matrix[(type1, type2)][1] += 1
    return matrix


def prob_locs(loc1, loc2, type1, type2, comatrix, scale=4):
    """
    This gives the prior distribution for Pr(s1 = loc1, s2 = loc2).

    This is formulated as:

    Pr(s1 = loc1, s2 = loc2) = sum Pr(s1 = loc1, s2 = loc2 | C) Pr(C)
                                C

    where C is a binary variable that is True if objects of type 1 and type 2
    co-occurs and False if not. The Pr(C) can be estimated from comatrix.
    And the term Pr(s1 = loc1, s2 = loc2 | C) can be straightforwardly
    defined based on the cooccur function.
    """
    if (type1, type2) in comatrix:
        counts = comatrix[(type1, type2)]
    elif (type2, type1) in comatrix:
        counts = comatrix[(type2, type1)]
    else:
        raise ValueError("No counts for {}, {}".format(type1, type2))
    n_cooccur, n_not_cooccur = counts
    if n_cooccur == 0 and n_not_cooccur == 0:
        return 1  # always return 1; independent

    # Pr(s1 = loc1, s2 = loc2 | C = True); Let's leave the normalization for later.
    pr_t = indicator(cooccur(loc1, loc2, scale=scale))\
            * (n_cooccur / (n_cooccur + n_not_cooccur))
    pr_f = indicator(not cooccur(loc1, loc2, scale=scale))\
            * (n_not_cooccur / (n_cooccur + n_not_cooccur))
    return pr_t + pr_f


def build_factor(locations, obj1, obj2, comatrix, scale=4):
    """
    Given a grid_map (GridMap), two objects, obj1, obj2 (Object),
    and a cooccurrence matrix, return a TabularDistribution that
    represents Pr(s_obj1, s_obj2) over the given locations

    The object is expected to have attribute "class"
    """
    type1 = obj1["class"]
    type2 = obj2["class"]
    weights = []
    for grid_loc1 in locations:

        objstate1 = LocObjState(obj1.id, type1, {"loc": grid_loc1})

        for grid_loc2 in locations:
            objstate2 = LocObjState(obj2.id, type2, {"loc": grid_loc2})

            setting = [(svar(obj1.id), objstate1),
                       (svar(obj2.id), objstate2)]
            prob = prob_locs(grid_loc1, grid_loc2, type1, type2,
                             comatrix, scale=scale)
            weights.append((setting, prob))
    factor = TabularDistribution([svar(obj1.id), svar(obj2.id)],
                                 weights)
    return factor


# A little test
def test():
    matrix = cooccur_matrix(robothor_scene_names("Train"), scale=12)
    pprint(["{}:{}".format(k,matrix[k]) for k in sorted(matrix, key=matrix.get)])


if __name__ == "__main__":
    test()
