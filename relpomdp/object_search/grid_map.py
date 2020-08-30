import pomdp_py
import copy
from relpomdp.object_search.relation import *
from relpomdp.object_search.state import *
from relpomdp.object_search.action import *
from relpomdp.object_search.condition_effect import *
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
        self._grid_map = grid_map

    def valid_motions(self, robot_pose):
        """
        Returns a set of MotionAction(s) that are valid to
        be executed from robot pose (i.e. they will not bump
        into obstacles). The validity is determined under
        the assumption that the robot dynamics is deterministic.
        """
        return set(self._legal_actions[robot_pose[:2]])

    def path_between(self, position1, position2, return_actions=True):
        """Note that for the return_actions=True case to return reasonable
        actions, the motion actions scheme needs to be `xy`, i.e. along the axes"""
        # Finds a path between position1 and position2.
        # Uses the Dijkstra's algorithm.
        V = set({(x,y)    # all valid positions
                 for x in range(self._grid_map.width) 
                 for y in range(self._grid_map.length)
                 if self._grid_map.within_bounds((x,y))})
        position1 = position1[:2]  # If it is robot pose then it has length 3.
        S = set({})
        d = {v:float("inf")
             for v in V
             if v != position1}
        d[position1] = 0
        prev = {position1: None}
        while len(S) < len(V):
            diff_set = V - S
            v = min(diff_set, key=lambda v: d[v])
            S.add(v)
            neighbors = self.get_neighbors(v)
            for w in neighbors:
                motion_action = neighbors[w]
                cost_vw = motion_action.distance_cost
                if d[v] + cost_vw < d[w[:2]]:
                    d[w[:2]] = d[v] + cost_vw
                    prev[w[:2]] = (v, motion_action)

        # Return a path
        path = []
        pair = prev[position2[:2]]
        if pair is None:
            if not return_actions:
                path.append(position2)
        else:
            while pair is not None:
                position, action = pair
                if return_actions:
                    # Return a sequence of actions that moves the robot from position1 to position2.
                    path.append(action)
                else:
                    # Returns the grid cells along the path
                    path.append(position)
                pair = prev[position]
        return list(reversed(path))
    

    def get_neighbors(self, robot_pose):
        neighbors = {MoveEffect.move_by(robot_pose, action.motion):action
                     for action in self.valid_motions(robot_pose)}
        return neighbors
