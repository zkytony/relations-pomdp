# This POMDP begins with no map

import pomdp_py
from relpomdp.home2d.agent.tests.test import wait_for_action
from relpomdp.home2d.agent.nk_agent import NKAgent, FakeSLAM
from relpomdp.home2d.tasks.common.sensor import Laser2DSensor
from relpomdp.home2d.agent.visual import NKAgentViz
from relpomdp.home2d.domain.maps.build_map import random_world
from relpomdp.home2d.domain.env import Home2DEnvironment
from relpomdp.home2d.agent.transition_model import Pickup
from relpomdp.oopomdp.framework import Objstate, OOState
import copy

def make_world():
    robot_id = 0
    init_robot_pose = (0, 0, 0)
    init_state, grid_map = random_world(6, 6, 3,
                                        ["Kitchen", "Office", "Office"],
                                        objects={"Office": {"Computer": (1, (1,1))},
                                                 "Kitchen": {"Pepper": (1, (1,1))},
                                                 "Bathroom": {"Toilet": (1, (1,1))},
                                                 "Corridor": {"Salt": (1, (1,1))}},
                                        robot_id=robot_id, init_robot_pose=init_robot_pose)
    env = Home2DEnvironment(robot_id,
                            grid_map,
                            init_state)
    return env


def test_pomdp_nk():
    env = make_world()
    robot_id = env.robot_id
    init_robot_pose = env.robot_state["pose"]
    nk_agent = NKAgent(robot_id, init_robot_pose)
    fake_slam = FakeSLAM(Laser2DSensor(robot_id,
                                       fov=90, min_range=1,
                                       max_range=3, angle_increment=0.1))

    target_class = "Salt"
    target_id = list(env.ids_for(target_class))[0]

    # Uniform belief over free spaces and a layer of frontier
    frontier = nk_agent.grid_map.frontier()
    print(frontier)

    viz = NKAgentViz(nk_agent,
                     env,
                     {},
                     res=30,
                     controllable=True,
                     img_path="../../domain/imgs")
    viz.on_init()

    for i in range(100):
        # Visualize
        viz.on_loop()
        viz.on_render()

        action = wait_for_action(viz)
        reward = env.state_transition(action, execute=True)
        print("[{}]     action: {}     reward: {}".format(i, str(action.name), str(reward)))

        robot_pose = env.state.object_states[robot_id]["pose"]  # should come from agent's belief
        fake_slam.update(nk_agent.grid_map, robot_pose, env)



if __name__ == "__main__":
    test_pomdp_nk()
