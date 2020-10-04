import pomdp_py
import copy
from relpomdp.home2d.domain.relation import *
from relpomdp.home2d.domain.state import *
from relpomdp.home2d.domain.action import *
from relpomdp.home2d.domain.condition_effect import *
from relpomdp.home2d.utils import objstate
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

    def compute_legal_motions(self, all_motion_actions):
        """Returns a map from (x,y) to legal motion actions"""
        legal_actions = {}  # 
        for x in range(self.width):
            for y in range(self.length):
                so = objstate("PoseObject", pose=(x,y))
                motion_actions = set(all_motion_actions)
                for wall_id in self.walls:
                    wall = self.walls[wall_id]
                    if touch_N(so, wall):
                        motion_actions.remove(MoveN)
                    elif touch_S(so, wall):
                        motion_actions.remove(MoveS)
                    elif touch_E(so, wall):
                        motion_actions.remove(MoveE)
                    elif touch_W(so, wall):
                        motion_actions.remove(MoveW)
                legal_actions[(x,y)] = motion_actions
        return legal_actions
