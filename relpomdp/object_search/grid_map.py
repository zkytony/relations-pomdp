import pomdp_py
import copy
from relpomdp.object_search.relation import *
from relpomdp.object_search.state import *
from relpomdp.object_search.action import *

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
            return self.xy_to_room[position]
        else:
            return None
            
    def within_bounds(self, position):
        if not (position[0] >= 0 and position[0] < self.width\
                and position[1] >= 0 and position[1] < self.length):
            return False
        return True

class MotionPolicy:
    def __init__(self, grid_map, all_motion_actions={MoveN, MoveS, MoveE, MoveW}):
        # Compute the valid motion actions at every location
        self._legal_actions = {}  # maps from (x,y) to legal motion actions
        for x in range(grid_map.width):
            for y in range(grid_map.length):
                objstate = PoseState("PoseObject", {"pose":(x,y)})
                motion_actions = set(all_motion_actions)
                for wall_id in grid_map.walls:
                    wall = grid_map.walls[wall_id]
                    if touch_N(objstate, wall):
                        motion_actions.remove(MoveN)
                    elif touch_S(objstate, wall):
                        motion_actions.remove(MoveS)
                    elif touch_E(objstate, wall):
                        motion_actions.remove(MoveE)
                    elif touch_W(objstate, wall):
                        motion_actions.remove(MoveW)
                self._legal_actions[(x,y)] = motion_actions

    def valid_motions(self, robot_pose):
        """
        Returns a set of MotionAction(s) that are valid to
        be executed from robot pose (i.e. they will not bump
        into obstacles). The validity is determined under
        the assumption that the robot dynamics is deterministic.
        """
        return set(self._legal_actions[robot_pose[:2]])
