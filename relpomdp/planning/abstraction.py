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
import time


class RoomAttr(AbstractAttribute):
    """This is an abstract attribute for Pose"""
    def __init__(self, room_name):
        super().__init__("room", room_name)
    @property
    def room_name(self):
        return self.value
    def copy(self):
        return RoomAttr(self.value)
    def reverse_image(self, grid_map):
        # Returns a set of Pose attributes
        room = grid_map.rooms[self.room_name]
        return [Pose(loc) for loc in room.locations]
    @classmethod
    def abstractify(self, pose_attribute, grid_map):
        if type(pose_attribute) == tuple:
            room_name = grid_map.room_of(pose_attribute[:2])
        else:  # Pose class
            room_name = grid_map.room_of(pose_attribute.value[:2])
        return RoomAttr(room_name)
