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
        self._show_img(img)
        return img

    def highlight(self, locations, color=(53, 190, 232)):
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

    def _show_img(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        pygame.surfarray.blit_array(self._display_surf, img)
        pygame.display.flip()
