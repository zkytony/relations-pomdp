# Visualize agent, allows control, visualize its map versus the true map

import pygame
import pomdp_py
from relpomdp.home2d.domain.visual import Home2DViz, lighter
from relpomdp.home2d.domain.maps.build_map import random_world
from relpomdp.home2d.domain.env import Home2DEnvironment
from relpomdp.home2d.agent.nk_agent import NKAgent
import pomdp_py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2
import time

class NKAgentViz(Home2DViz):

    def __init__(self, nkagent, env, colors, img_path="imgs",
                 res=30, fps=30, controllable=False):
        super().__init__(env, colors, img_path=img_path,
                         res=res, fps=fps, controllable=controllable)

        self._nkagent = nkagent
        self._room_colors = {}

    def on_init(self):
        super().on_init()
        plt.ion()
        self._fig, (self._ax1, self._ax2) = plt.subplots(1,2, figsize=(12,5))
        plt.show(block=False)

    def _make_legend(self, ax, used_colors):
        patches = []
        for objclass in used_colors:
            color = used_colors[objclass]
            patches.append(mpatches.Patch(color=np.array(color)/255.0, label=objclass))
        ax.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    def on_render(self, belief=None, **kwargs):
        # Renders the true world. Then plot agent's world
        img_world = super().on_render()

        # self._fig.clear()
        img = self.make_agent_view(self._res,
                                   range_sensor=kwargs.get("range_sensor", None))
        used_colors = {}

        if belief is not None:
            circle_drawn = {}
            for objid in belief.object_beliefs:
                belief_obj = belief.object_beliefs[objid]
                objclass = belief.object_beliefs[objid].mpe().objclass
                if objclass.lower() == "robot":
                    continue

                color = self._colors.get(objclass, (128, 128, 128))
                NKAgentViz.draw_object_belief(img, self._res, belief_obj, color,
                                              circle_drawn=circle_drawn)
                used_colors[objclass] = color

        # rotate 90 deg CCW to match the pygame display
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        self._ax2.imshow(img, interpolation='none')
        # These must happen after imshow
        self._ax2.set_aspect("equal")
        self._make_legend(self._ax2, used_colors)

        img_world_copy = cv2.flip(img_world, 1)  # flip horizontally
        img_world_copy = cv2.rotate(img_world_copy, cv2.ROTATE_90_COUNTERCLOCKWISE)
        self._ax1.imshow(img_world_copy, interpolation='none')

        self._fig.canvas.draw()
        self._fig.canvas.flush_events()
        return img, img_world

    @staticmethod
    def draw_object_belief(img, r, belief, color,
                           circle_drawn={}, use_alpha=True):
        """
        circle_drawn: map from pose to number of times drawn;
            Used to determine sizxe of circle to draw at a location
        """
        radius = int(round(r / 2))
        size = r // 3
        last_val = -1
        hist = belief.get_histogram()
        for state in reversed(sorted(hist, key=hist.get)):
            if last_val != -1:
                color = lighter(color, 1-hist[state]/last_val)
            if np.mean(np.array(color) / np.array([255, 255, 255])) < 0.999:
                tx, ty = state['pose']
                if (tx,ty) not in circle_drawn:
                    circle_drawn[(tx,ty)] = 0
                circle_drawn[(tx,ty)] += 1

                cv2.circle(img, (ty*r+radius,
                                 tx*r+radius), size//circle_drawn[(tx,ty)], color, thickness=-1)
                last_val = hist[state]

                if last_val <= 0:
                    break


    def make_agent_view(self, r, range_sensor=None, visualize_room_segmentation=False):
        # Preparing 2d array
        w, l = self._env.width, self._env.length
        state = self._env.state
        rx, ry, rth = self._env.robot_state["pose"]

        agent_map = self._nkagent.grid_map
        frontier = agent_map.frontier()

        # Creating image
        img = np.full((w*r,l*r,3), 255, dtype=np.int32)
        for x in range(w):
            for y in range(l):
                # Draw free space; If this exists in robot's partial map,
                # draw it with white; Otherwise, dark
                if (x,y) in agent_map.free_locations:
                    room_color = (255, 255, 255)
                    boundary_color = (0, 0, 0)
                elif (x,y) in frontier:
                    room_color = (250, 243, 55)
                    boundary_color = (245, 241, 142)
                else:
                    room_color = (66, 66, 66)
                    boundary_color = (66, 66, 66)

                if visualize_room_segmentation:
                    room_name = agent_map._location_to_room.get((x,y), None)
                    if room_name is not None:
                        if room_name not in self._room_colors:
                            self._room_colors[room_name] =\
                                pomdp_py.util.random_unique_color(self._room_colors.values())
                            self._room_colors[room_name] = pomdp_py.util.hex_to_rgb(self._room_colors[room_name])
                        room_color = self._room_colors[room_name]

                cv2.rectangle(img, (y*r, x*r), (y*r+r, x*r+r),
                              room_color, -1)
                # Draw boundary
                cv2.rectangle(img, (y*r, x*r), (y*r+r, x*r+r),
                              boundary_color, 1, 8)

                # Draw indication of field of view
                if range_sensor is not None:
                    if (x,y) in agent_map.free_locations or (x,y) in frontier:
                        # if (rx, ry) == (1, 0) and (x,y) == (5,5):
                        #     import pdb; pdb.set_trace()
                        if range_sensor.within_range((rx, ry, rth), (x,y), grid_map=agent_map):
                            print((rx, ry, rth), (x,y))
                            fov_color = (101, 213, 247)
                            padding = r//5
                            cv2.rectangle(img, (y*r+padding, x*r+padding), (y*r+r-padding, x*r+r-padding),
                                          fov_color, -1)
        self.render_walls(self._nkagent.grid_map.walls, img, r)

        # draw robot
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
