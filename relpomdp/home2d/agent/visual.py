# Visualize agent, allows control, visualize its map versus the true map

import pygame
from relpomdp.home2d.domain.visual import Home2DViz
from relpomdp.home2d.domain.maps.build_map import random_world
from relpomdp.home2d.domain.env import Home2DEnvironment
from relpomdp.home2d.agent.nk_agent import NKAgent
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time

class NKAgentViz(Home2DViz):

    def __init__(self, nkagent, env, colors, img_path="imgs",
                 res=30, fps=30, controllable=False):
        self._nkagent = nkagent
        super().__init__(env, colors, img_path=img_path,
                         res=res, fps=fps, controllable=controllable)

    def on_init(self):
        super().on_init()
        plt.ion()
        plt.show(block=False)

    def on_render(self):
        # Renders the true world. Then plot agent's world
        super().on_render()

        plt.clf()
        fig = plt.gcf()
        ax = plt.gca()
        img = self.make_agent_view(self._res)
        # rotate 90 deg CCW to match the pygame display
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        ax.imshow(img, interpolation='none')
        # These must happen after imshow
        ax.set_aspect("equal")
        # ax.invert_yaxis()

        fig.canvas.draw()
        fig.canvas.flush_events()

    def make_agent_view(self, r):
        # Preparing 2d array
        w, l = self._env.width, self._env.length
        state = self._env.state

        # Creating image
        img = np.full((w*r,l*r,3), 255, dtype=np.int32)
        for x in range(w):
            for y in range(l):
                # Draw free space; If this exists in robot's partial map,
                # draw it with white; Otherwise, dark
                if (x,y) in self._nkagent.grid_map.free_locations:
                    room_color = (255, 255, 255)
                    boundary_color = (0, 0, 0)
                else:
                    room_color = (66, 66, 66)
                    boundary_color = (66, 66, 66)
                cv2.rectangle(img, (y*r, x*r), (y*r+r, x*r+r),
                              room_color, -1)
                # Draw boundary
                cv2.rectangle(img, (y*r, x*r), (y*r+r, x*r+r),
                              boundary_color, 1, 8)
        self.render_walls(self._nkagent.grid_map.walls, img, r)

        # draw robot
        rx, ry, rth = self._env.robot_state["pose"]
        NKAgentViz.draw_robot(img, rx*r, ry*r, rth, r, r*0.85)

        return img



def unittest():
    robot_id = 0
    init_robot_pose = (0, 0, 0)
    init_state, grid_map = random_world(10, 10, 4,
                                        ["Office", "Office", "Kitchen", "Bathroom"],
                                        robot_id=robot_id, init_robot_pose=init_robot_pose)
    agent = NKAgent(init_robot_pose)
    env = Home2DEnvironment(robot_id,
                            grid_map,
                            init_state)
    viz = NKAgentViz(agent,
                     env,
                     {},
                     res=30,
                     controllable=True)
    viz.on_init()
    viz.on_execute()

if __name__ == "__main__":
    unittest()
