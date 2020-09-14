import pomdp_py
import copy
from relpomdp.oopomdp.abstraction import AbstractAttribute
from relpomdp.object_search.world_specs.build_world import small_map1
from relpomdp.object_search.state import *
from relpomdp.object_search.agent import *
from relpomdp.object_search.env import *
from relpomdp.object_search.sensor import *
from relpomdp.object_search.reward_model import RewardModel
from relpomdp.object_search.tests.worlds import *
from relpomdp.object_search.world_specs.build_world import *
import relpomdp.oopomdp.framework as oopomdp
from relpomdp.planning.subgoal import Subgoal
from relpomdp.object_search.abstraction import RoomAttr
import time

class ReachRoomSubgoal(Subgoal):
    """
    This subgoal is achieved when the robot reaches a particular room.
    """
    def __init__(self, ids, room_type, grid_map):
        self.ids = ids
        self.grid_map = grid_map
        self.room_type = room_type
        super().__init__("Reach-%s" % room_type)
        
    def achieve(self, state, action):
        # Achieves the goal when the robot is at the center of mass of the room
        robot_id = self.ids["Robot"]
        robot_state = state.object_states[robot_id]
        room_attr = RoomAttr.abstractify(robot_state.pose, self.grid_map)
        room = self.grid_map.rooms[room_attr.room_name]
        return room.room_type == self.room_type\
            and robot_state.pose[:2] == room.center_of_mass

    def fail(self, state, action):
        return isinstance(action, Pickup)

    def trigger_success(self, robot_state, action, observation):
        room_attr = RoomAttr.abstractify(robot_state.pose, self.grid_map)
        subgoal = SearchRoomSubgoal(self.ids, room_attr.room_name, self.grid_map)
        return subgoal

class SearchRoomSubgoal(Subgoal):
    """
    This subgoal is achieved when the target object is found
    within the room being searched. It fails when either the
    robot steps outside of the room or 
    """
    def __init__(self, ids, room_name, grid_map):
        self.ids = ids
        self.room_name = room_name
        self.grid_map = grid_map
        super().__init__("Search-%s" % room_name)

    def achieve(self, state, action):
        robot_id = self.ids["Robot"]
        robot_state = state.object_states[robot_id]
        room_attr = RoomAttr.abstractify(robot_state.pose, self.grid_map)
        if room_attr.room_name != self.room_name:
            return False
        for objid in self.ids["Target"]:
            if state.object_states[objid]["is_found"]:
                return True
        return False

    def fail(self, state, action):
        robot_id = self.ids["Robot"]        
        robot_state = state.object_states[robot_id]
        if isinstance(action, Pickup):
            for objid in self.ids["Target"]:
                objstate = state.object_states[objid]
                if not (objstate.pose == robot_state.pose[:2]\
                        and not objstate["is_found"]):
                    return True
        return False

def interpret_subgoal(string, **kwargs):
    ids = kwargs.get("ids", None)
    grid_map = kwargs.get("grid_map", None)
    if string.startswith("Reach"):
        room_type = string.split("_")[1]
        subgoal = ReachRoomSubgoal(ids, room_type, grid_map)
    else:
        room_name = string.split("_")[1]
        subgoal = SearchRoomSubgoal(ids, room_name, grid_map)
    return subgoal
