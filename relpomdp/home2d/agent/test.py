import pygame
from relpomdp.home2d.domain.visual import Home2DViz
from relpomdp.home2d.domain.maps.build_map import random_world
from relpomdp.home2d.domain.env import Home2DEnvironment
from relpomdp.home2d.agent.nk_agent import NKAgent, FakeSLAM
from relpomdp.home2d.agent.visual import NKAgentViz
from relpomdp.home2d.tasks.common.sensor import Laser2DSensor
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time

def wait_for_action(viz, timeout=10):
    action = None
    time_start = time.time()
    print("Waiting for action...")
    while action is None:
        for event in pygame.event.get():
            action = viz.on_event(event)
            if action is not None:
                break
        if time.time() - time_start > timeout:
            raise ValueError("No action after %.1fs" % timeout)
    return action


def main():
    robot_id = 0
    init_robot_pose = (0, 0, 0)
    init_state, grid_map = random_world(10, 10, 4,
                                        ["Office", "Office", "Kitchen", "Bathroom"],
                                        robot_id=robot_id, init_robot_pose=init_robot_pose)
    env = Home2DEnvironment(robot_id,
                            grid_map,
                            init_state)
    agent = NKAgent(init_robot_pose)
    fake_slam = FakeSLAM(Laser2DSensor(robot_id,
                                       fov=90, min_range=1,
                                       max_range=3, angle_increment=0.5))

    viz = NKAgentViz(agent,
                     env,
                     {},
                     res=30,
                     controllable=True)
    viz.on_init()

    for i in range(100):
        # Visualize
        viz.on_loop()
        viz.on_render()

        action = wait_for_action(viz)
        reward = env.state_transition(action, execute=True)
        print("[{}]     action: {}     reward: {}".format(i, str(action.name), str(reward)))

        robot_pose = env.state.object_states[robot_id]["pose"]  # should come from agent's belief
        fake_slam.update(agent.grid_map, robot_pose, env)

if __name__ == "__main__":
    main()
