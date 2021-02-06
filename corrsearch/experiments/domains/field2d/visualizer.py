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
from corrsearch.experiments.domains.field2d.problem import Field2D
from corrsearch.utils import overlay

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
        self.problem = problem
        self._bg_path = bg_path
        self._res = config.get("res", 30)   # resolution
        self._linewidth = config.get("linewidth", 8)
        super().__init__(problem)

    def visualize(self, state, belief=None):
        """
        Args:
            state (JointState): full state of the world
            belief (JointBelief): Belief of the agent
        """
        img = self._make_gridworld_image(self._res, state)
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
            bgimg = cv2.resize(bgimg, (l*r, w*r))
            img = overlay(img, bgimg, opacity=1.0, pos=(50,50))
            # img = cv2.addWeighted(img, 1.0, bgimg, 0.0, 0)

        for x in range(w):
            for y in range(l):
                # Draw boundary
                cv2.rectangle(img, (y*r, x*r), (y*r+r, x*r+r),
                              (0, 0, 0), 1, self._linewidth)

        return img


if __name__ == "__main__":
    problem = Field2D((10, 10))
    viz = Field2DViz(problem, "./testimg.png")
    img = viz.visualize(None)
    cv2.imshow("TT", img)
    cv2.waitKey(1000)
