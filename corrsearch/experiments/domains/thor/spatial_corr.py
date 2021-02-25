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
from corrsearch.utils import indicator, euclidean_dist
from pprint import pprint
import itertools
import random


def uniform(*args):
    return True

def nearby(point1, point2, radius=2):
    return euclidean_dist(point1, point2) <= radius

def not_nearby(point1, point2, radius=2):
    return not nearby(point1, point2, radius=radius)


class SpatialJointDist(JointDist):
    """
    A spatial joint distribution is a wrapper of a joint_dist
    that can sample or compute probability for object locations
    at different grid sizes.
    """
    def __init__(self, joint_dist, grid_sizes, sampling_grid_size=0.25):
        """
        joint_dist (JointDist) is a distribution over variables for
            object locations and each variable has a range at some
            (likely coarse) grid size (granularity level)

        grid_sizes (dict): Maps from variable name to grid_size
            that the variable's values are defined over in the joint_dist.

        sampling_grid_size (float): The grid size that we want to sample
            object locations and compute probabilities at

        Note: We assume that sampling_grid_size is smaller than any of the
            grid sizes. AND the ratio is an integer
        """
        self.joint_dist = joint_dist
        self.grid_sizes = grid_size
        self.sampling_grid_size = sampling_grid_size
        super().__init__(self.joint_dist.variables)

    def prob(self, values):
        """
        Args:
            values (dict): Mapping from variable to value.
                Does not have to specify the value for every variable
        """
        raise NotImplementedError

    def sample(self, rnd=random):
        raise NotImplementedError

    def marginal(self, variables, observation=None):
        """Performs marignal inference,
        produce a joint distribution over `variables`,
        given evidence (if supplied)"""
        raise NotImplementedError

    def valrange(self, var):
        """Returns an enumerable that contains the possible values
        of the given variable var"""
        vals_fine = set()
        vals_coarse = self.joint_dist.valrange(var)
        ratio = self.grid_sizes[var] // self.sampling_grid_size
        for si_coarse in vals_coarse:
            coarse_x, coarse_y = si_coarse.loc
            for fine_x in (coarse_x * ratio, (coarse_x+1) * ratio):
                for fine_y in (coarse_y * ratio, (coarse_y+1) * ratio):
                    si_fine = LocObjState(si_coarse.id, si_coarse["class"],
                                          {"loc": (fine_x, fine_y)})
                    vals_fine.add(si_fine)
        return vals_fine

if __name__ == "__main__":
    test()
