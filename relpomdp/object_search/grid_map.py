import pomdp_py
import copy
from relpomdp.object_search.relation import *
from relpomdp.object_search.state import *
from relpomdp.object_search.action import *

class GridMap:
    def __init__(self, width, length, walls):
        """walls (dict): Map from objid to WallState."""
        self.width = width
        self.length = length
        self.walls = walls
            
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
