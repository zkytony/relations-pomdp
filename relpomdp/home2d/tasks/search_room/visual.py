import pygame
from relpomdp.home2d.domain.visual import Home2DViz, lighter
from relpomdp.home2d.tasks.search_room.search_room_task import Stop
from relpomdp.home2d.tasks.search_item.visual import SearchItemViz
import numpy as np
import cv2


class SearchRoomViz(Home2DViz):
    def __init__(self, env, colors, img_path="imgs",
                 res=30, fps=30, controllable=False):
        super().__init__(env, colors, img_path="imgs",
                         res=30, fps=30, controllable=False)
        self._key_action_map[pygame.K_SPACE] = Stop
        self._last_belief = None

    def render_env(self, display_surf):
        # Draw belief
        r = self._res  # Not radius!
        img = super().render_env(display_surf)
        if self._last_belief is not None:
            SearchItemViz.draw_belief(img, self._last_belief, r, r//3, self._colors)
        # Draw robot (again)
        rx, ry, rth = self._env.robot_state["pose"]
        Home2DViz.draw_robot(img, rx*r, ry*r, rth, r, r*0.85)            
        return img

    def update(self, belief):
        """
        Update the visualization after there is new real action and observation
        and updated belief.
        """
        self._last_belief = belief

    @staticmethod
    def draw_belief(img, belief, r, size, colors):
        """belief (OOBelief)"""
        SearchItemViz.draw_belief(img, belief, r, size, colors)
