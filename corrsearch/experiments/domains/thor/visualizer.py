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
from corrsearch.utils import overlay, lighter, lighter_with_alpha, cv2shape
from corrsearch.experiments.domains.thor.topo_maps.build_topo_map import draw_topo
from corrsearch.experiments.domains.thor.thor import thor_object_poses

class ThorViz(Visualizer):

    def __init__(self, problem, **config):
        super().__init__(problem)

        # Actually grab all the detectable object poses
        object_poses = {}
        for objid in problem.objects:
            object_type = problem.objects[objid]["class"]
            if object_type not in object_poses:
                object_poses[object_type] = set()

            thor_instance_poses = thor_object_poses(problem.env.controller, object_type)
            for thor_objid in thor_instance_poses:
                thor_x, thor_y, thor_z = thor_instance_poses[thor_objid]
                objloc = problem.grid_map.to_grid_pos(thor_x, thor_z,
                                                      grid_size=problem.grid_size)
                object_poses[object_type].add(objloc)
        self.object_poses = object_poses

        self._res = config.get("res", 30)   # resolution
        self._linewidth = config.get("linewidth", 1)
        self._save = config.get("save", False)

        self._bg_path = config.get("bg_path", None)
        self._visuals = []

        self.on_init()

    def _make_gridworld_image(self, r):
        # Preparing 2d array
        grid_map = self.problem.grid_map
        w, l = grid_map.width, grid_map.length

        # Make an image of grids
        if self._bg_path is not None:
            bgimg = cv2.imread(self._bg_path, cv2.IMREAD_UNCHANGED)
            bgimg = cv2.resize(bgimg, (w*r, l*r))
            img = overlay(img, bgimg, opacity=1.0)

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

    def visualize(self, state, belief=None, action=None, sensor=None):
        img = self._make_gridworld_image(self._res)

        # Draw topo map, if there is one
        if hasattr(self.problem.env, "topo_map"):
            topo_map = self.problem.env.topo_map
            img = draw_topo(img, topo_map.to_json(),
                            self.problem.grid_map, self.problem.env.grid_size, self)

        robot_pose = state[self.problem.robot_id]["pose"]
        if sensor is not None:
            img = self.draw_fov(img, robot_pose, sensor)

        # # Draw where the objects are
        # for object_type in self.object_poses:
        #     if object_type == "robot":
        #         continue
        #     objid = self.problem.scene_info.objid_for_type(object_type)
        #     for obj_x, obj_y in self.object_poses[object_type]:
        #         r = self._res
        #         cv2.rectangle(img, (obj_y*r, obj_x*r),
        #                       (obj_y*r+r, obj_x*r+r),
        #                       lighter(self.get_color(objid)[:3], 0.5), -1)

        # Draw belief (only about target)
        target_id = self.problem.target_id
        if belief is not None:
            target_color = self.get_color(target_id, alpha=0.8)
            img = self.draw_object_belief(img, belief.obj(target_id),
                                          target_color)

        # Draw robot
        x, y, th = robot_pose
        color = self.get_color(self.problem.robot_id)
        img = self.draw_robot(img, x, y, th,
                              color=color)

        self.show_img(img)

        # Save visualization. Grab the screenshot from THOR as well.
        if self._save:
            controller = self.problem.env.controller
            event = controller.step(action="Pass")
            event = controller.step(action="Pass")
            frame = event.frame
            controller.step(action="ToggleMapView")
            event = controller.step(action="Pass")
            event = controller.step(action="Pass")
            frame_topdown = event.frame
            controller.step(action="ToggleMapView")
            self._visuals.append({"img": img,
                                  "frame": frame,
                                  "frame_topdown": frame_topdown})
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
        self.show_img(img)
        return img

    def gridworld_img(self):
        return self._make_gridworld_image(self._res)

    def draw_fov(self, img, robot_pose, sensor):
        size = self._res
        radius = int(round(size / 2))

        # Draw FOV
        grid_map = self.problem.grid_map
        for x in range(self.problem.grid_map.width):
            for y in range(self.problem.grid_map.length):
                if sensor.in_range((x,y), robot_pose):
                    img = cv2shape(img, cv2.circle,
                                   (y*self._res+radius,
                                    x*self._res+radius), radius,
                                   (200, 200, 36), thickness=-1, alpha=0.75)
        return img

    def get_color(self, objid, default=(220, 150, 10, 255), alpha=1.0):
        color = self.problem.obj(objid).get("color", default)
        if len(color) == 3:
            color = color + [int(round(alpha*255))]
        color = tuple(color)
        return color

    def show_img(self, img):
        """
        Internally, the img origin (0,0) is top-left (that is the opencv image),
        so +x is right, +z is down.
        But when displaying, to match the THOR unity's orientation, the image
        is flipped, so that in the displayed image, +x is right, +z is up.
        """
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        img = cv2.flip(img, 1)  # flip horizontally
        pygame.surfarray.blit_array(self._display_surf, img)
        pygame.display.flip()

    def draw_robot(self, img, x, y, th, color=(255, 150, 0)):
        """Note: agent by default (0 angle) looks in the +z direction in Unity,
        which corresponds to +y here. That's why I'm multiplying y with cos."""
        size = self._res
        x *= self._res
        y *= self._res

        radius = int(round(size / 2))
        shift = int(round(self._res / 2))
        cv2.circle(img, (y+shift, x+shift), radius, color, thickness=2)

        endpoint = (y+shift + int(round(shift*math.cos(th))),
                    x+shift + int(round(shift*math.sin(th))))
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

                img = cv2shape(img, cv2.rectangle,
                               (ty*self._res,#radius,
                                tx*self._res),#radius),
                               (ty*self._res+self._res,#radius,
                                tx*self._res+self._res),#radius),
                               # size//circle_drawn[(tx,ty)],
                               color, thickness=-1, alpha=color[3]/255)
                last_val = hist[state]
                if last_val <= 0:
                    break
        return img

    def save_visuals(self, trial_path):
        visual_path = os.path.join(trial_path, "visuals")
        os.makedirs(visual_path, exist_ok=True)
        for step, visual in enumerate(self._visuals):
            print("Saving visual for step %d" % step)
            for kind in self._visuals[step]:
                img = self._visuals[step][kind]
                if kind == "img":
                    img = cv2.flip(img, 1)  # flip horizontally
                    img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)  # rotate 90deg clockwise
                self._visuals[step][kind] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cv2.imwrite(os.path.join(visual_path, "img-%d.png" % step), visual["img"])
            cv2.imwrite(os.path.join(visual_path, "frame-%d.png" % step), visual["frame"])
            cv2.imwrite(os.path.join(visual_path, "frame_topdown-%d.png" % step), visual["frame_topdown"])
