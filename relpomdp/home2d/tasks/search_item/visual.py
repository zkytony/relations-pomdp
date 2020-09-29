import pygame
from relpomdp.home2d.domain.visual import Home2DViz, lighter
from relpomdp.home2d.tasks.search_item.search_item_task import Pickup
import numpy as np
import cv2


class SearchItemViz(Home2DViz):
    def __init__(self, env, colors, img_path="imgs",
                 res=30, fps=30, controllable=False):
        super().__init__(env, colors, img_path="imgs",
                         res=30, fps=30, controllable=False)
        self._key_action_map[pygame.K_SPACE] = Pickup
        self._last_belief = None

    def render_env(self, display_surf):
        # Draw belief
        r = self._res  # Not radius!
        draw_funcs = []
        draw_funcs_args = {}
        if self._last_belief is not None:
            args = ([self._last_belief, r//3, self._colors], {})
            draw_funcs = [SearchItemViz.draw_belief]
            draw_funcs_args = [args]

        img = super().render_env(display_surf, draw_funcs=draw_funcs,
                                 draw_funcs_args=draw_funcs_args)
        return img

    def update(self, belief):
        """
        Update the visualization after there is new real action and observation
        and updated belief.
        """
        self._last_belief = belief

    @staticmethod
    def draw_belief(img, r, belief, size, colors):
        """belief (OOBelief)"""
        radius = int(round(r / 2))

        circle_drawn = {}  # map from pose to number of times drawn

        if type(belief) == dict:
            object_beliefs = belief
        else:
            object_beliefs = belief.object_beliefs

        for objid in object_beliefs:
            if isinstance(object_beliefs[objid], dict):
                hist = object_beliefs[objid]
            else:
                if object_beliefs[objid].random().objclass.lower() == "robot":
                    continue
                hist = object_beliefs[objid].get_histogram()
                
            if objid in colors:
                color = colors[objid]
            else:
                color = (255, 165, 0)

            last_val = -1
            count = 0
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

                    count +=1
                    if last_val <= 0:
                        break
        return img
