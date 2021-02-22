import unittest
import time
import random
import math
from corrsearch.models import *
from corrsearch.utils import *
from corrsearch.objects import ObjectState, JointState
from corrsearch.experiments.domains.thor.conversion import *
from corrsearch.experiments.domains.thor.problem import *
from corrsearch.experiments.domains.thor.grid_map import *
from corrsearch.experiments.domains.thor.thor import *
from corrsearch.experiments.domains.thor.visualizer import *
from corrsearch.experiments.domains.thor.transition import *

def setUp():
    config = {
        "scene_name": "FloorPlan_Train1_1",
        "width": 400,
        "height": 400,
        "grid_size": 0.25
    }
    controller = launch_controller(config)
    grid_map = convert_scene_to_grid_map(controller, config["scene_name"], config["grid_size"])
    controller.grid_size = config["grid_size"]
    return controller, grid_map

def test_thor_visualize():
    controller, grid_map = setUp()
    robot_id = 0
    init_robot_pose = (*random.sample(grid_map.free_locations, 1)[0], 0.0)
    state = JointState({robot_id: RobotState(robot_id, {"pose": init_robot_pose,
                                                        "energy":0.0})})
    region = grid_map.free_region(*init_robot_pose[:2])

    problem = ThorSearch(robot_id)
    problem.grid_map = grid_map

    viz = ThorViz(problem)
    viz.visualize(state)
    time.sleep(3)
    controller.stop()

def test_thor_viz_highlight():
    controller, grid_map = setUp()

    robot_id = 0
    init_robot_pose = (*random.sample(grid_map.free_locations, 1)[0], 0.0)
    region = grid_map.free_region(*init_robot_pose[:2])

    problem = ThorSearch(robot_id)
    problem.grid_map = grid_map

    viz = ThorViz(problem)
    viz.highlight(region)
    time.sleep(2)
    controller.stop()


def test_thor_moving():
    controller, grid_map = setUp()
    controller.step("ToggleMapView")

    robot_id = 0
    problem = ThorSearch(robot_id)
    problem.grid_map = grid_map

    thor_pose2d = thor_agent_pose2d(controller)
    print(thor_pose2d)
    grid_loc2d = grid_map.to_grid_pos(thor_pose2d[0], thor_pose2d[1],
                                          grid_size=controller.grid_size)
    init_robot_pose = (*grid_loc2d, to_rad(thor_pose2d[2]))
    print(init_robot_pose)
    state = JointState({robot_id: RobotState(robot_id, {"pose": init_robot_pose,
                                                        "energy":0.0})})
    viz = ThorViz(problem)
    viz.visualize(state)
    time.sleep(10)

    trans_model = DetRobotTrans(robot_id, grid_map)
    forward = Move((1.0, 0.0), "forward")
    backward = Move((-1.0, 0.0), "backward")
    left = Move((0.0, -math.pi/4), "left")
    right = Move((0.0, math.pi/4), "right")

    next_state = JointState({robot_id: trans_model.sample(state, forward)})
    viz.visualize(state)
    time.sleep(10)


if __name__ == "__main__":
    # test_thor_visualize()
    test_thor_moving()
