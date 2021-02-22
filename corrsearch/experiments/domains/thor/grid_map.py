import pomdp_py
import copy
import math
import json
import sys
from corrsearch.objects import *
from corrsearch.utils import remap

class Wall(ObjectState):
    def __init__(self, id, pose, direction):
        """direction can be 'H' or 'V'"""
        super().__init__(id, "wall",
                         {"pose": pose,
                          "direction": direction})
    def copy(self):
        return WallState(self.id, tuple(self["pose"]), self["direction"])

    @property
    def segment(self):
        """
        Even though the wall's pose are in integers, because
        the wall is placed at the edge of a cell, its effective
        pose is slightly shifted resulting in a segment
        """
        if self.direction == "V":
            wall_pose1 = (self["pose"][0]+0.5, self["pose"][1]+0.5)
            wall_pose2 = (self["pose"][0]+0.5, self["pose"][1]-0.5)
        else:
            wall_pose1 = (self["pose"][0]+0.5, self["pose"][1]+0.5)
            wall_pose2 = (self["pose"][0]-0.5, self["pose"][1]+0.5)
        wall_seg = [wall_pose1, wall_pose2]
        return wall_seg

    def intersect(self, src, dst):
        """Returns true if the wall intersects with a light ray
        shooting from src to dst (2d points)"""
        return utils.intersect(self.segment, (src, dst))

    @property
    def pose(self):
        return self["pose"]

    def cells_touching(self):
        """Returns the location of the two cells
        that is touching this wall. Note that a vertical
        wall is on the right side of a cell, and a horizontal
        wall is on the top side of a cell"""
        if self.direction == "V":
            x, y = self["pose"]
            return ((x,y), (x+1,y))
        else:
            x, y = self["pose"]
            return ((x,y), (x,y+1))


class GridMap:
    """
    A grid map is a collection of locations, walls, and some locations have obstacles.
    The coordinate convention differs from THOR. Horizontal axis is x, vertical axis is y.

    Note that the coordinates in grid map starts from (0,0) to (width-1,length-1).
    This differs from THOR which could have negative coordinates
    """

    def __init__(self, width, length, walls, obstacles, name="grid_map",
                 ranges_in_thor=None):
        """
        xpos (list): list of x coordinates for free cells
        ypos (list): list of y coordinates for free cells
        walls (dict): Map from objid to WallState.
        obstacles (set): a set of locations for the obstacles
        ranges_in_thor (tuple): A tuple (thor_gx_range, thor_gy_range)
            where thor_gx_range are the min max range for grid x coordiates in thor
            where thor_gy_range are the min max range for grid y coordiates in thor
            Note that thor grid coordinates do not originate in (0,0) but at some other
                point determined by Unity.
        """
        self.width = width
        self.length = length
        self.walls = walls

        self.obstacles = obstacles
        self.free_locations = {(x,y) for x in range(width)
                               for y in range(length)
                               if (x,y) not in obstacles}

        self.name = name
        self.ranges_in_thor = ranges_in_thor

    def within_bounds(self, position):
        if not (position[0] >= 0 and position[0] < self.width\
                and position[1] >= 0 and position[1] < self.length):
            return False
        return True

    def containers(self, container_type):
        if container_type == "Room":
            return {name:self.rooms[name].to_state()
                    for name in self.rooms}

    def legal_motions_at(self, x, y, all_motion_actions, permitted_locations=None):
        """
        permitted_locations (set): A set of (x,y) locations besides what is in
            that we allow the robot to move to. If None, `self.free_locations`
            will be used.
        """
        motion_actions = set(all_motion_actions)
        legal_actions = set()
        if permitted_locations is None:
            permitted_locations = self.free_locations

        # Compute maximum expected length of travel
        max_traj_len = float("-inf")
        for a in motion_actions:
            dx, dy, dth = a.motion
            traj = (x,y), (x+dx, y+dy)
            max_traj_len = max(max_traj_len, math.sqrt(dx**2 + dy**2))

        # Find walls that are nearby
        nearby_walls = set()
        for wall_id in self.walls:
            p1, p2 = self.walls[wall_id].segment
            if euclidean_dist(p1, (x,y)) <= max_traj_len\
               or euclidean_dist(p2, (x,y)) <= max_traj_len:
                nearby_walls.add(wall_id)

        # Compute legal motions
        for a in motion_actions:
            dx, dy, dth = a.motion
            if x+dx < 0:
                continue
            if y+dy < 0:
                continue
            if (x+dx, y+dy) not in permitted_locations:
                continue
            legal = True
            for wall_id in nearby_walls:
                if self.walls[wall_id].intersect((x,y), (x+dx, y+dy)):
                    legal = False
                    break
            if legal:
                legal_actions.add(a)
        return legal_actions


    def compute_legal_motions(self, all_motion_actions):
        """Returns a map from (x,y) to legal motion actions"""
        legal_actions = {}  #
        for x in range(self.width):
            for y in range(self.length):
                legal_actions[(x,y)] = self.legal_motions_at(x, y, all_motion_actions)
        return legal_actions


    def to_thor_pos(self, x, y, grid_size=None):
        """
        Given a point (x,y) in the grid map, convert it to (x,z) in
        the THOR coordinte system (grid size is accounted for).
        If grid_size is None, will return the integers
        for the corresponding coordinate.
        """
        # Note that y is z in Unity
        thor_gx_range = self.ranges_in_thor[0]
        thor_gy_range = self.ranges_in_thor[1]
        thor_gx = remap(x, 0, self.width, thor_gx_range[0], thor_gx_range[1])
        thor_gy = remap(y, 0, self.width, thor_gy_range[0], thor_gy_range[1])
        if grid_size is not None:
            return (thor_gx * grid_size, thor_gy * grid_size)
        else:
            return (thor_gx, thor_gy)
