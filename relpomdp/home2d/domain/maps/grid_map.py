import pomdp_py
import copy
from relpomdp.home2d.domain.relation import *
from relpomdp.home2d.domain.state import *
from relpomdp.home2d.domain.action import *
from relpomdp.home2d.domain.condition_effect import *
from relpomdp.oopomdp.framework import Objstate
from relpomdp.utils_geometry import intersect, euclidean_dist
import math
import sys

class GridMap:
    def __init__(self, width, length, walls, rooms):
        """
        walls (dict): Map from objid to WallState.
        rooms (list): A list of rooms
            (includes both rooms and corridors)"""
        self.width = width
        self.length = length
        self.walls = walls
        self.rooms = {r.name:r for r in rooms}

        # Create a mapping from location to room
        self.xy_to_room = {}
        for name in self.rooms:
            room = self.rooms[name]
            for x,y in room.locations:
                self.xy_to_room[(x,y)] = room.name

        # all locations are free
        self.free_locations = {(x,y) for x in range(width)
                               for y in range(length)}

    def room_of(self, position):
        if position in self.xy_to_room:
            return self.rooms[self.xy_to_room[position]]
        else:
            return None

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
        so = Objstate("PoseObject", pose=(x,y))
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
