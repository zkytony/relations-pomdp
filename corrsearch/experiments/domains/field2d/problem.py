"""
This is a 2D domain that captures the gist of
what we want for our problem.
"""

import pomdp_py
from corrsearch.models.problem import *

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
    def __init__(self, dim):
        self.dim = dim
