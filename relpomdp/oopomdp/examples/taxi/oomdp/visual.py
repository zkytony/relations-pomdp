# Visualization of a TAXI instance using pygame
#
# the wall is on the right side of the grid cell.

import pygame
import cv2
import math
import numpy as np
import random
from pomdp_py import util
from relpomdp.oopomdp.examples.taxi.env import WallState, DestinationState
from relpomdp.oopomdp.examples.taxi.actions import *

#### Visualization through pygame ####
class TaxiViz:

    def __init__(self, env,
                 res=30, fps=30, controllable=False):
        self._env = env

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
        objstates_at_pose = {}
        for objid in state.object_states:
            pose = state.object_states[objid]["pose"]
            if pose not in objstates_at_pose:
                objstates_at_pose[pose] = []
            objstates_at_pose[pose].append(state.object_states[objid])

        # Creating image
        img = np.full((w*r,l*r,3), 255, dtype=np.int32)
        for x in range(w):
            for y in range(l):
                # Draw free space
                cv2.rectangle(img, (y*r, x*r), (y*r+r, x*r+r),
                              (255, 255, 255), -1)                
                if (x,y) in objstates_at_pose:
                    for objstate in objstates_at_pose[(x,y)]:
                        if objstate.objclass == WallState.CLASS:
                            if objstate.direction == "H":
                                # draw a line on the top side of the square
                                cv2.line(img, (y*r+r, x*r), (y*r+r, x*r+r),
                                         (0, 0, 0), 6)
                            else:
                                # draw a line on the right side of the square
                                cv2.line(img, (y*r, x*r+r), (y*r+r, x*r+r),
                                         (0, 0, 0), 6)
                                
                        elif objstate.objclass == DestinationState.CLASS:
                            cv2.rectangle(img, (y*r, x*r), (y*r+r, x*r+r),
                                              (199, 42, 85), 4)
                # Draw boundary
                cv2.rectangle(img, (y*r, x*r), (y*r+r, x*r+r),
                              (0, 0, 0), 1, 8)                    
        return img
    
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
    def draw_taxi(img, x, y, res, size, color=(255,150,0)):
        radius = int(round(size / 2))        
        shift = int(round(res / 2))
        cv2.circle(img, (y+shift, x+shift), radius, color, thickness=2)
        cv2.circle(img, (y+shift, x+shift), radius//2, color, thickness=2)        

    @staticmethod
    def draw_passenger(img, x, y, res, size, color=(10,180,40)):
        radius = int(round(size / 2))
        shift = int(round(res / 2))
        cv2.circle(img, (y+shift, x+shift), radius, color, thickness=-1)
        cv2.circle(img, (y+shift, x+shift), radius//2, (10,60,20), thickness=-1)        

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
            elif event.key == pygame.K_RETURN:
                action = Dropoff()

            if action is None:
                return

            if self._controllable:
                reward = self._env.state_transition(action, execute=True)
                print("taxi state: %s" % str(self._env.state.taxi_state))
                print("psgr state: %s" % str(self._env.state.passenger_state))                
                print("     action: %s" % str(action.name))
                print("     reward: %s" % str(reward))
                print("------------")
            return action

    def on_loop(self):
        self._playtime += self._clock.tick(self._fps) / 1000.0
        
    def on_render(self):
        # self._display_surf.blit(self._background, (0, 0))
        self.render_env(self._display_surf)
        rx, ry = self._env.state.taxi_state["pose"]
        fps_text = "FPS: {0:.2f}".format(self._clock.get_fps())
        pygame.display.set_caption("Taxi(%.2f,%.2f) | %s" %
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
        # draw robot, a circle and a vector
        img = np.copy(self._img)
        rx, ry = self._env.state.taxi_state["pose"]
        px, py = self._env.state.passenger_state["pose"]
        r = self._res  # Not radius!
        TaxiViz.draw_taxi(img, rx*r, ry*r, r, r*0.85)
        TaxiViz.draw_passenger(img, px*r, py*r, r, r*0.75)

        # In numpy image array, (0,0) is on top-left. But
        # we want to visualize it so that it's on bottom-left,
        # matching the definition in Taxi. So we just flip the image.
        img = cv2.flip(img, 1)  # flip horizontally 
        pygame.surfarray.blit_array(display_surf, img)
