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
import itertools

def cooccur(grid_loc1, grid_loc2, scale=4):
    """Given two grid locations, return True if they fall in the same
    bigger grid that is `scale` times bigger."""
    supergrid_loc1 = grid_loc1[0] // scale, grid_loc1[1] // scale
    supergrid_loc2 = grid_loc2[0] // scale, grid_loc2[1] // scale
    return supergrid_loc1 == supergrid_loc2

def cooccur_matrix_instance_counts(scenes, grid_size=0.25, scale=4):
    """
    Returns a cooccurrence matrix (mapping from [i,j] -> (#cooccur, #not-cooccur))
    for all pairs of object types in the scenes, as well as
    a mapping that maps from object type to number of occurrences of the type
    """
    matrix = {}
    scene_infos = {scene_name : load_scene_info(scene_name)
                   for scene_name in scenes}
    for scene_name, scene_info in scene_infos.items():
        objtypes = scene_info.obj_types()
        counts = {}
        for type1, type2 in itertools.product(objtypes, objtypes):
            if (type2, type1) in counts:
                counts[(type1, type2)] = counts[(type2, type1)]
                continue
            counts[(type1, type2)] = 0
            for objid1 in scene_info.pomdp_objids(type1):
                thor_pose1 = scene_info.thor_obj_pose2d(objid1)
                grid_loc1 = (thor_pose1[0] // grid_size, thor_pose1[1] // grid_size)
                for objid2 in scene_info.pomdp_objids(type2):
                    if objid1 == objid2:
                        continue
                    thor_pose2 = scene_info.thor_obj_pose2d(objid2)
                    grid_loc2 = (thor_pose2[0] // grid_size, thor_pose2[1] // grid_size)
                    if cooccur(grid_loc1, grid_loc2, scale=scale):
                        counts[(type1, type2)] += 1

        for type1, type2 in counts:
            if (type1, type2) not in matrix:
                matrix[(type1, type2)] = 0
                matrix[(type2, type1)] = 0
            matrix[(type1, type2)] += counts[(type1, type2)]
            matrix[(type2, type1)] = matrix[(type1, type2)]

    final_matrix = {}  # {type -> {type -> counts}}
    for type1, type2 in matrix:
        final_matrix[type1] = final_matrix.get(type1, {})
        final_matrix[type1][type2] = matrix[(type1,type2)]

    occurrences = {}  # maps from object type to number of occurrences of the type
    for scene_name, scene_info in scene_infos.items():
        objtypes = scene_info.obj_types()
        for objtype in objtypes:
            occurrences[objtype] = occurrences.get(objtype, 0) \
                                   + len(scene_info.pomdp_objids(objtype))
    return final_matrix, occurrences


def cooccur_matrix_scene_counts(scenes, grid_size=0.25, scale=4):
    """
    Returns a cooccurrence matrix (mapping from [i,j] -> #scenes_where_there_is_coocurrence
    for all pairs of object types in the scenes,
    as well as a mapping that maps from object type to number of SCENES the type occurred
    """
    matrix = {}
    scene_infos = {scene_name : load_scene_info(scene_name)
                   for scene_name in scenes}
    for scene_name, scene_info in scene_infos.items():
        objtypes = scene_info.obj_types()
        cooc = {}
        for type1, type2 in itertools.product(objtypes, objtypes):
            if (type2, type1) in cooc:
                cooc[(type1, type2)] = cooc[(type2, type1)]
                continue
            coocurring = False
            for objid1, objid2 in itertools.product(scene_info.pomdp_objids(type1),
                                                    scene_info.pomdp_objids(type2)):
                if objid1 == objid2:
                    continue
                thor_pose1 = scene_info.thor_obj_pose2d(objid1)
                grid_loc1 = (thor_pose1[0] // grid_size, thor_pose1[1] // grid_size)
                thor_pose2 = scene_info.thor_obj_pose2d(objid2)
                grid_loc2 = (thor_pose2[0] // grid_size, thor_pose2[1] // grid_size)
                if cooccur(grid_loc1, grid_loc2, scale=scale):
                    coocurring = True
                    break
            cooc[(type1, type2)] = coocurring

        for type1, type2 in cooc:
            if (type1, type2) not in matrix:
                matrix[(type1, type2)] = 0
                matrix[(type2, type1)] = 0
            if cooc[(type1, type2)]:
                matrix[(type1, type2)] += 1
            matrix[(type2, type1)] = matrix[(type1, type2)]

    final_matrix = {}  # {type -> {type -> counts}}
    for type1, type2 in matrix:
        final_matrix[type1] = final_matrix.get(type1, {})
        final_matrix[type1][type2] = matrix[(type1,type2)]

    occurrences = {}  # maps from object type to number of SCENES the type occurred
    for scene_name, scene_info in scene_infos.items():
        objtypes = scene_info.obj_types()
        for objtype in objtypes:
            occurrences[objtype] = occurrences.get(objtype, 0)
            if len(scene_info.pomdp_objids(objtype)) > 0:
                occurrences[objtype] += 1
    return final_matrix, occurrences



# Build conditional distributions directly; All that we need is
# the conditionals.
class SpatialCorrDist(JointDist):

    def __init__(self, scene_info, locations, comatrix, occurrences,
                 target_id, scale=4, cooccur_type="scene_counts"):
        self.scene_info = scene_info
        self.comatrix = comatrix
        self.target_id = target_id
        self.locations = locations
        self.scale = scale
        self.cooccur_type = cooccur_type
        self.occurs = occurrences

    def marginal(self, variables, observation=None):
        """We only account for the case Pr(si | starget)"""
        if len(variables) != 1:
            raise ValueError("Only accept one variable.")
        if observation is None or len(observation) != 1:
            raise ValueError("observation (starget) must not be None / must only contain target")
        var_corrobj = variables[0]
        var_target, starget = list(observation.items())[0]
        assert svar(self.target_id) == var_target

        corrobj = self.scene_info.obj(id_from_svar(var_corrobj))
        targetobj = self.scene_info.obj(self.target_id)

        weights = []
        for corrobj_loc in self.locations:
            state_corrobj = LocObjState(corrobj.id, corrobj["class"], {"loc": corrobj_loc})
            prob = self._factor(corrobj_loc, corrobj, starget.loc, targetobj)
            setting = [(var_corrobj, state_corrobj)]
            weights.append((setting, prob))
        return TabularDistribution([var_corrobj], weights)

    def _factor(self, loc, obj, cond_loc, cond_obj):
        """Given an object is at cond_loc,
        return the probability that another object is at loc"""
        type_obj = obj["class"]
        type_cond_obj = cond_obj["class"]

        if self.cooccur_type == "scene_counts":
            # More than half of the scenes where type_obj appears, type_obj and type_cond_obj
            # spatially cooccur --> call that spatially correlated; spatially correlated
            # here means spatailly co-occurring.
            spatially_correlated =\
                (self.comatrix[type_obj][type_cond_obj] / self.occurs[type_obj]) >= self.occurs[type_obj] * 0.5
            if spatially_correlated:
                return indicator(cooccur(loc, cond_loc))
            else:
                # Not spatially correlated; independent.
                return 1.0

        elif self.cooccur_type == "instance_counts":
            no_cooccur = self.comatrix[type_obj][type_cond_obj]
            if cooccur(loc, cond_loc):
                return no_cooccur / self.occurs[type_obj]
            else:
                return 1 - no_cooccur / self.occurs[type_obj]

        else:
            raise ValueError("Unrecognized coocurrence type {}".format(self.cooccur_type))


# A little test
def test():
    matrix, occurs = cooccur_matrix_scene_counts(robothor_scene_names("Train"), scale=12)
    pprint(matrix)


if __name__ == "__main__":
    test()
