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

    def _make_gridworld_image(self, r, state):
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
        self._render_walls(grid_map.walls, img, r)
        return img

    def _render_walls(self, walls, img, r, wall_color=(0,0,0,255)):
        """r == resolution"""
        # Draw walls
        walls_at_pose = {}
        for objid in walls:
            wall = walls[objid]
            x, y = wall["pose"]
            if wall.direction == "H":
                # draw a line on the top side of the square
                cv2.line(img, (y*r+r, x*r), (y*r+r, x*r+r),
                         wall_color, 6)
            else:
                # draw a line on the right side of the square
                cv2.line(img, (y*r, x*r+r), (y*r+r, x*r+r),
                         wall_color, 6)

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
        img = self._make_gridworld_image(self._res, state)

        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        # import pdb; pdb.set_trace()
        pygame.surfarray.blit_array(self._display_surf, img)

        # rx, ry, th = state[self.problem.robot_id]["pose"]
        # fps_text = "FPS: {0:.2f}".format(self._clock.get_fps())
        # pygame.display.set_caption("robot_pose(%.2f,%.2f,%.2f) | %s" %
        #                            (rx, ry, th,
        #                             fps_text))
        pygame.display.flip()
        return img
