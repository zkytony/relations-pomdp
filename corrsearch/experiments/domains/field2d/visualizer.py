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
from corrsearch.utils import overlay, lighter, lighter_with_alpha, cv2shape
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

    def get_color(self, objid, default=(128, 128, 128, 255), alpha=1.0):
        color = self.problem.obj(objid).get("color", default)
        if len(color) == 3:
            color = color + [int(round(alpha*255))]
        color = tuple(color)
        return color

    def visualize(self, state, belief=None):
        """
        Args:
            state (JointState): full state of the world
            belief (JointBelief): Belief of the agent
        """
        # Makes the background image with grid overlay
        img = self._make_gridworld_image(self._res, state)

        print(state)

        # Place objects
        for objid in range(len(state.object_states)):
            if objid != self.problem.robot_id:
                x, y = state[objid]["loc"]
                dim = self.problem.obj(objid).get("dim", (1,1))
                obj_img_path = self.problem.obj(objid).get("obj_img_path", None)
                color = self.get_color(objid, default=(30, 10, 200, 255))
                img = self.draw_object(img, x, y, dim, color=color, obj_img_path=obj_img_path,
                                       is_target=objid==self.problem.target_id)

        # Draw belief (only about target)
        if belief is not None:
            target_id = self.problem.target_id
            target_color = self.get_color(target_id, alpha=0.8)
            img = self.draw_object_belief(img, belief.obj(target_id),
                                          target_color)

        # Draw robot
        x, y, th = state[self.problem.robot_id]["pose"]
        color = self.get_color(self.problem.robot_id)
        img = self.draw_robot(img, x, y, th,
                              color=color)

        # Render; Need to rotate and flip so that pygame displays
        # the image in the same way as opencv where (0,0) is top left.
        img_render = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)  # rotate 90deg clockwise
        img_render = cv2.flip(img_render, 0)  # flip horizontally
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
                    obj_img_path=None, add_bg=False, is_target=False):
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
        else:
            # No image. draw a square.
            radius = int(round(self._res / 2))
            shift = int(round(self._res / 2))
            cv2.circle(img, (starty+shift, startx+shift), radius,
                       color, thickness=-1)
            if is_target:
                cv2.circle(img, (starty+shift, startx+shift), radius//2,
                           (30, 200, 30, 255), thickness=-1)

        return img

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

    def draw_object_belief(self, img, belief, color,
                           circle_drawn=None):
        """
        circle_drawn: map from pose to number of times drawn;
            Used to determine size of circle to draw at a location
        """
        if circle_drawn is None:
            circle_drawn = {}
        radius = int(round(self._res / 2))
        size = self._res // 3
        last_val = -1
        hist = belief.get_histogram()
        for state in reversed(sorted(hist, key=hist.get)):
            if last_val != -1:
                color = lighter_with_alpha(color, 1-hist[state]/last_val)

            if len(color) == 4:
                stop = color[3]/255 < 0.1
            else:
                stop = np.mean(np.array(color[:3]) / np.array([255, 255, 255])) < 0.999

            if not stop:
                tx, ty = state['loc']
                if (tx,ty) not in circle_drawn:
                    circle_drawn[(tx,ty)] = 0
                circle_drawn[(tx,ty)] += 1

                img = cv2shape(img, cv2.circle,
                               (ty*self._res+radius,
                                tx*self._res+radius), size//circle_drawn[(tx,ty)],
                               color, thickness=-1, alpha=color[3]/255)
                last_val = hist[state]
                if last_val <= 0:
                    break
        return img
