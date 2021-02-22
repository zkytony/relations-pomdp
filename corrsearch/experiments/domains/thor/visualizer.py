"""
Visualizer
"""
import pygame
import cv2
import numpy as np
import math
import random
import time
import os
from corrsearch.models.visualizer import Visualizer


class ThorViz(Visualizer):

    def __init__(self, problem, **config):
        super().__init__(problem)
        self._res = config.get("res", 30)   # resolution
        self._linewidth = config.get("linewidth", 1)
        self.on_init()

    def _make_gridworld_image(self, r):
        # Preparing 2d array
        grid_map = self.problem.grid_map
        w, l = grid_map.width, grid_map.length

        # Make an image of grids
        img = np.full((w*r, l*r, 4), 255, dtype=np.uint8)
        for x in range(w):
            for y in range(l):
                if (x,y) in grid_map.obstacles:
                    cv2.rectangle(img, (y*r, x*r), (y*r+r, x*r+r),
                                  (40, 31, 3), -1)
                # Draw boundary
                cv2.rectangle(img, (y*r, x*r), (y*r+r, x*r+r),
                              (0, 0, 0), self._linewidth)
        return img

    @property
    def img_width(self):
        return self.problem.grid_map.width * self._res

    @property
    def img_height(self):
        return self.problem.grid_map.length * self._res


    def on_init(self):
        """pygame init"""
        pygame.init()  # calls pygame.font.init()
        # init main screen and background
        self._display_surf = pygame.display.set_mode((self.img_width,
                                                      self.img_height),
                                                     pygame.HWSURFACE)
        self._background = pygame.Surface(self._display_surf.get_size()).convert()
        self._clock = pygame.time.Clock()

        # Font
        self._myfont = pygame.font.SysFont('Comic Sans MS', 30)
        self._running = True

    def visualize(self, state, belief=None, action=None):
        img = self._make_gridworld_image(self._res)

        # Draw robot
        x, y, th = state[self.problem.robot_id]["pose"]
        color = self.get_color(self.problem.robot_id)
        img = self.draw_robot(img, x, y, th,
                              color=color)

        self._show_img(img)
        return img

    def highlight(self, locations, color=(53, 190, 232)):
        """Color the grid cells of the given locations.
        TODO: Doesn't render anything else right now"""
        r = self._res
        img = self._make_gridworld_image(r)
        for x, y in locations:
            cv2.rectangle(img, (y*r, x*r), (y*r+r, x*r+r),
                          color, -1)
            # Draw boundary
            cv2.rectangle(img, (y*r, x*r), (y*r+r, x*r+r),
                          (0, 0, 0), self._linewidth)
        self._show_img(img)
        return img

    def get_color(self, objid, default=(128, 128, 128, 255), alpha=1.0):
        color = self.problem.obj(objid).get("color", default)
        if len(color) == 3:
            color = color + [int(round(alpha*255))]
        color = tuple(color)
        return color

    def _show_img(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        img = cv2.flip(img, 1)  # flip horizontally
        pygame.surfarray.blit_array(self._display_surf, img)
        pygame.display.flip()

    def draw_robot(self, img, x, y, th, color=(255, 150, 0)):
        size = self._res
        x *= self._res
        y *= self._res

        radius = int(round(size / 2))
        shift = int(round(self._res / 2))
        cv2.circle(img, (y+shift, x+shift), radius, color, thickness=2)

        endpoint = (y+shift + int(round(shift*math.sin(th))),
                    x+shift + int(round(shift*math.cos(th))))
        cv2.line(img, (y+shift,x+shift), endpoint, color, 2)
        return img
