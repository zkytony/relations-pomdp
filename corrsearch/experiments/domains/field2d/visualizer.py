"""
Again, builds upon my 2D pygame grid world.
 Why not?
"""
import pygame
import cv2
import numpy as np
import math
import random
import time
import os
from corrsearch.models.visualizer import Visualizer
from corrsearch.utils import overlay
from corrsearch.objects import ObjectState, JointState

class Field2DViz(Visualizer):

    def __init__(self, problem, bg_path=None, **config):
        """
        The visualization is done through an image, which is displayed through pygame.
        A limitation, that I was not satisfied with, was displaying:

        - images with transparent background
        - rendering things with transparency

        I probably want to use a better image representation than numpy array
        maintained all by myself.
        """
        super().__init__(problem)
        self._bg_path = bg_path
        self._res = config.get("res", 30)   # resolution
        self._linewidth = config.get("linewidth", 1)
        self.on_init()

    @property
    def img_width(self):
        return self.problem.dim[0] * self._res

    @property
    def img_height(self):
        return self.problem.dim[1] * self._res

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

    def visualize(self, state, belief=None):
        """
        Args:
            state (JointState): full state of the world
            belief (JointBelief): Belief of the agent
        """
        # Makes the background image with grid overlay
        img = self._make_gridworld_image(self._res, state)

        # Place objects
        for objid in range(len(state.object_states)):
            x, y = state[objid]["loc"]
            dim = state[objid].get("dim", (1,1))
            color = state[objid].get("base_color", (128, 128, 128, 255))
            obj_img_path = state[objid].get("obj_img_path", None)
            img = self.draw_object(img, x, y, dim, color=color, obj_img_path=obj_img_path)

        # Render
        img_render = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)  # rotate 90deg clockwise
        img_render = cv2.flip(img_render, 1)  # flip horizontally
        img_render = cv2.cvtColor(img_render, cv2.COLOR_RGBA2BGR)
        pygame.surfarray.blit_array(self._display_surf, img_render)

        rx, ry, th = state[self.problem.robot_id]["pose"]
        fps_text = "FPS: {0:.2f}".format(self._clock.get_fps())
        pygame.display.set_caption("robot_pose(%.2f,%.2f,%.2f) | %s" %
                                   (rx, ry, th,
                                    fps_text))
        pygame.display.flip()
        return img

    def _make_gridworld_image(self, r, state):
        """The image will have WxL dimension.
        The W stands for width (horizontal, i.e. number of columns),
        and H stands for height (vertical, i.e. number of rows)
        """
        w, l = self.problem.dim

        # Make an image of grids
        img = np.full((l*r, w*r, 4), 255, dtype=np.uint8)
        if self._bg_path is not None:
            bgimg = cv2.imread(self._bg_path, cv2.IMREAD_UNCHANGED)
            bgimg = cv2.resize(bgimg, (w*r, l*r))
            img = overlay(img, bgimg, opacity=1.0)

        for x in range(w):
            for y in range(l):
                # Draw boundary
                cv2.rectangle(img, (x*r, y*r), (x*r+r, y*r+r),
                              (0, 0, 0), self._linewidth)

        return img

    def draw_object(self, img, x, y, dim,
                    color=(128, 128, 128, 255),
                    obj_img_path=None, add_bg=False):
        """
        Draws an object at location (x,y) that takes up a footprint of rectangle
        of dimensions `dim` (w,l). The object image will be overlayed. The
        `res` is the resolution. The width w refers to the number of columns (horizontal),
        and the length l is vertical.
        """
        # First, color the footprint
        startx = max(0, x*self._res)
        starty = max(0, y*self._res)
        endx = min(img.shape[1], (x+dim[0])*self._res)
        endy = min(img.shape[0], (y+dim[1])*self._res)
        if add_bg:
            cv2.rectangle(img, (startx, starty), (endx, endy), color, thickness=-1)
        # Then, overlay the image
        if obj_img_path is not None:
            objimg = cv2.imread(obj_img_path, cv2.IMREAD_UNCHANGED)
            objimg = cv2.resize(objimg, (dim[0]*self._res, dim[1]*self._res))
            overlay(img, objimg, opacity=1.0, pos=(startx, starty))
        return img
