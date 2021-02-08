"""
This is a 2D domain that captures the gist of
what we want for our problem.
"""

import yaml
import pomdp_py
from corrsearch.models.problem import *
from corrsearch.objects.template import Object
from corrsearch.experiments.domains.field2d.detector import *

class Field2D(SearchProblem):
    """
    A Field2D problem is defined by:
    - a NxM grid of locations
    - objects, distributions
    - object detectors that have different:
      - capability
      - range
      - noise
    - the reward depends on success and also robot's energy level
    """
    def __init__(self, dim, robot_id, objects, detectors,
                 locations=None, **kwargs):
        self.dim = dim
        self.name = kwargs.get("name", "field2d")
        if locations is None:
            locations = [(x,y) for x in range(dim[0]) for y in range(dim[1])]
