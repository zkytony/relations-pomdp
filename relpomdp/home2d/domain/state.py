import numpy as np
import pomdp_py
import relpomdp.oopomdp.framework as oopomdp
from relpomdp.home2d.utils import objstate
import relpomdp.home2d.utils as utils
import copy

# Defines the basic states in a home2d task. Besides the wall,
# everything else can be represented with the oopomdp.ObjectState
# and oopomdp.OOState classes directly.

class WallState(oopomdp.ObjectState):
    def __init__(self, pose, direction):
        """direction can be 'H' or 'V'"""
        super().__init__("Wall",
                         {"pose": pose,
                          "direction": direction})
    def copy(self):
        return WallState(tuple(self["pose"]), self.direction)
    @property
    def direction(self):
        return self["direction"]
    
    def intersect(self, src, dst):
        """Returns true if the wall intersects with a light ray
        shooting from src to dst (2d points)"""
        if self.direction == "V":
            wall_pose1 = (self["pose"][0]+0.5, self["pose"][1]+0.5)
            wall_pose2 = (self["pose"][0]+0.5, self["pose"][1]-0.5)                        
        else:
            wall_pose1 = (self["pose"][0]+0.5, self["pose"][1]+0.5)
            wall_pose2 = (self["pose"][0]-0.5, self["pose"][1]+0.5)
        wall_seg = [wall_pose1, wall_pose2]
        return utils.intersect(wall_seg, (src, dst))
    
    @property
    def pose(self):
        return self["pose"]
    
def unittest():
    assert WallState((0,1), "H") == WallState((0,1), "H")
    print("Passed")

if __name__ == "__main__":
    unittest()
