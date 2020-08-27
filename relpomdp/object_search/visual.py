# Visualization of a OBJECTSEARCH instance using pygame
#
# the wall is on the right side of the grid cell.

import pygame
import cv2
import math
import numpy as np
import random
import os
from pomdp_py import util
from relpomdp.object_search.state import *
from relpomdp.object_search.action import *

# colors
def lighter(color, percent):
    '''assumes color is rgb between (0, 0, 0) and (255, 255, 255)'''
    color = np.array(color)
    white = np.array([255, 255, 255])
    vector = white-color
    return color + vector * percent    

#### Visualization through pygame ####
class ObjectSearchViz:

    def __init__(self, env, colors, img_path="imgs",
                 res=30, fps=30, controllable=False):
        """colors: a mapping from object id to (R,G,B)."""
        self._env = env
        self._colors = colors
        self._img_path = img_path

        self._res = res
        self._img = self._make_gridworld_image(res)
        self._controllable = controllable
        self._running = False
        self._fps = fps
        self._playtime = 0.0


    def _make_gridworld_image(self, r):
        # Preparing 2d array
        w, l = self._env.width, self._env.length
        state = self._env.state

        # Creating image
        img = np.full((w*r,l*r,3), 255, dtype=np.int32)
        for x in range(w):
            for y in range(l):
                # Draw free space
                cv2.rectangle(img, (y*r, x*r), (y*r+r, x*r+r),
                              (255, 255, 255), -1)                
                # Draw boundary
                cv2.rectangle(img, (y*r, x*r), (y*r+r, x*r+r),
                              (0, 0, 0), 1, 8)
        self.render_walls(img, r)
        return img

    def render_walls(self, img, r):
        """r == resolution"""
        # Draw walls
        walls_at_pose = {}
        for objid in self._env.grid_map.walls:
            wall = self._env.grid_map.walls[objid]
            x, y = wall.pose
            if wall.direction == "H":
                # draw a line on the top side of the square
                cv2.line(img, (y*r+r, x*r), (y*r+r, x*r+r),
                         (0, 0, 0), 6)
            else:
                # draw a line on the right side of the square
                cv2.line(img, (y*r, x*r+r), (y*r+r, x*r+r),
                         (0, 0, 0), 6)
    
    @property
    def img_width(self):
        return self._img.shape[0]
    
    @property
    def img_height(self):
        return self._img.shape[1]

    @property
    def last_observation(self):
        return self._last_observation
    
    @staticmethod
    def draw_robot(img, x, y, res, size, color=(255,150,0)):
        radius = int(round(size / 2))        
        shift = int(round(res / 2))
        cv2.circle(img, (y+shift, x+shift), radius, color, thickness=2)
        cv2.circle(img, (y+shift, x+shift), radius//2, color, thickness=2)        

    @staticmethod
    def draw_object(img, x, y, res, size, color=(10,180,40), obj_img_path=None):
        if obj_img_path is not None:
            obj_img = cv2.imread(obj_img_path, cv2.IMREAD_COLOR)
            obj_img = cv2.rotate(obj_img, cv2.ROTATE_90_CLOCKWISE)  # rotate 90deg clockwise
            obj_img = cv2.cvtColor(obj_img, cv2.COLOR_BGR2RGB)
            obj_img = cv2.resize(obj_img, (res, res))
            w,l = obj_img.shape[:2]
            img[x:x+w, y:y+l] = obj_img
            # Draw boundary
            cv2.rectangle(img, (y, x), (y+res, x+res), (0, 0, 0), 1, 8)                    
        else:
            radius = int(round(size / 2))
            shift = int(round(res / 2))
            cv2.circle(img, (y+shift, x+shift), radius, color, thickness=-1)
            cv2.circle(img, (y+shift, x+shift), radius//2, lighter(color, 0.4), thickness=-1)

    # PyGame interface functions
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

    def on_event(self, event):
        # TODO: Keyboard control multiple robots
        if event.type == pygame.QUIT:
            self._running = False
        elif event.type == pygame.KEYDOWN:
            u = None  # control signal according to motion model
            action = None  # control input by user

            # odometry model
            if event.key == pygame.K_LEFT:
                action = MoveW
            elif event.key == pygame.K_RIGHT:
                action = MoveE
            elif event.key == pygame.K_UP:
                action = MoveN
            elif event.key == pygame.K_DOWN:
                action = MoveS
            elif event.key == pygame.K_SPACE:
                action = Pickup()

            if action is None:
                return

            if self._controllable:
                reward = self._env.state_transition(action, execute=True)
                print("robot state: %s" % str(self._env.robot_state))
                print("     action: %s" % str(action.name))
                print("     reward: %s" % str(reward))
                print("------------")
            return action

    def on_loop(self):
        self._playtime += self._clock.tick(self._fps) / 1000.0
        
    def on_render(self):
        # self._display_surf.blit(self._background, (0, 0))
        self.render_env(self._display_surf)
        rx, ry = self._env.robot_state["pose"]
        fps_text = "FPS: {0:.2f}".format(self._clock.get_fps())
        pygame.display.set_caption("Objectsearch(%.2f,%.2f) | %s" %
                                   (rx, ry,
                                    fps_text))
        pygame.display.flip() 
 
    def on_cleanup(self):
        pygame.quit()
 
    def on_execute(self):
        if self.on_init() == False:
            self._running = False
 
        while( self._running ):
            for event in pygame.event.get():
                self.on_event(event)
            self.on_loop()
            self.on_render()
        self.on_cleanup()

    def render_env(self, display_surf):
        img = np.copy(self._img)
        r = self._res  # Not radius!
        
        # Draw objects
        for objid in self._env.state.object_states:
            objstate = self._env.state.object_states[objid]
            if isinstance(objstate, RobotState):
                continue
            if objid in self._colors:
                color = self._colors[objid]
            else:
                color = (10,180,40)
            x, y = objstate.pose
            obj_img_path = os.path.join(self._img_path, "%s.png" %  objstate.objclass.lower())
            if not os.path.exists(obj_img_path):
                obj_img_path = None
            ObjectSearchViz.draw_object(img, x*r, y*r, r, r*0.75, color=color,
                                        obj_img_path=obj_img_path)

        # Draw robot
        rx, ry = self._env.robot_state["pose"]
        ObjectSearchViz.draw_robot(img, rx*r, ry*r, r, r*0.85)

        # Draw walls
        self.render_walls(img, r)        

        # In numpy image array, (0,0) is on top-left. But
        # we want to visualize it so that it's on bottom-left,
        # matching the definition in Taxi. So we just flip the image.
        img = cv2.flip(img, 1)  # flip horizontally 
        pygame.surfarray.blit_array(display_surf, img)
