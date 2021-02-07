"""
This is a 2D domain that captures the gist of
what we want for our problem.
"""

import yaml
import pomdp_py
from corrsearch.models.problem import *
from corrsearch.objects.template import Object

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
    def __init__(self, dim, **kwargs):
        self.dim = dim
        self.name = kwargs.get("name", "field2d")

def problem_parser(domain_file, dist_file):
    objects = []
    with open(domain_file) as f:
        spec = yaml.load(f, Loader=yaml.Loader)
        for i in range(len(spec["objects"])):
            objspec = spec["objects"][i]
            obj = Object(objspec["id"], objspec)
            objects.append(obj)


if __name__ == "__main__":
    problem_parser("./example_config.yaml", None)
