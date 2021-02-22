import unittest
import time
import random
from corrsearch.experiments.domains.thor.conversion import *
from corrsearch.experiments.domains.thor.problem import *
from corrsearch.experiments.domains.thor.grid_map import *
from corrsearch.experiments.domains.thor.thor import *
from corrsearch.experiments.domains.thor.visualizer import *

def test_thor_visualize():
    config = {
        "scene_name": "FloorPlan_Train1_1",
        "width": 400,
        "height": 400,
        "grid_size": 0.5
    }
    controller = launch_controller(config)
    grid_map = convert_scene_to_grid_map(controller, config["scene_name"], config["grid_size"])

    region = grid_map.free_region(*random.sample(grid_map.free_locations, 1)[0])

    problem = ThorSearch()
    problem.grid_map = grid_map

    viz = ThorViz(problem)
    viz.visualize(None)
    viz.highlight(region)
    time.sleep(5)

if __name__ == "__main__":
    test_thor_visualize()
