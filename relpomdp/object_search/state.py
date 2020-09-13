import numpy as np
import pomdp_py
import relpomdp.oopomdp.framework as oopomdp
from relpomdp.oopomdp.abstraction import AbstractAttribute
import relpomdp.object_search.utils as utils
import copy
    
# Object classes and attributes
class Pose(oopomdp.Attribute):
    def __init__(self, value):
        super().__init__("pose", value)
    def copy(self):
        assert type(self.value) == tuple
        return Pose(tuple(self.value))
    def __getitem__(self, index):
        assert type(self.value) == tuple        
        return self.value[index]

class PoseState(oopomdp.ObjectState):
    @property
    def pose(self):
        return self["pose"]
    
class WallState(PoseState):
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
            wall_pose1 = (self.pose[0]+0.5, self.pose[1]+0.5)
            wall_pose2 = (self.pose[0]+0.5, self.pose[1]-0.5)                        
        else:
            wall_pose1 = (self.pose[0]+0.5, self.pose[1]+0.5)
            wall_pose2 = (self.pose[0]-0.5, self.pose[1]+0.5)
        wall_seg = [wall_pose1, wall_pose2]
        return utils.intersect(wall_seg, (src, dst))

class ContainerState(oopomdp.ObjectState):
    def __init__(self, container_type, name, footprint):
        """name, e.g. Room"""
        super().__init__(container_type,
                         {"name": name,
                          "footprint": footprint})
    def copy(self):
        return self.__class__(self.category, self.name, tuple(self.footprint))
    @property
    def name(self):
        return self["name"]
    @property
    def category(self):
        return self.objclass        
    def copy(self):
        return self.__class__(self.container_type, tuple(self.footprint))
    @property
    def footprint(self):
        return self["footprint"]
    
class ItemState(PoseState):
    def __init__(self, name, pose, is_found=False):
        super().__init__(name,
                         {"pose": pose,
                          "is_found": is_found})                          
    @property
    def name(self):
        return self.objclass
    def copy(self):
        return self.__class__(self.name, tuple(self["pose"]), self["is_found"])
    @property
    def is_found(self):
        return self["is_found"]

class RobotState(PoseState):
    def __init__(self, pose, camera_direction):
        """
        pose (tuple): x,y,th
        objects_found (tuple): objects found
        camera_direction (string): direction of looking
        """
        super().__init__("Robot",
                         {"pose":pose,  # x,y,th
                          "camera_direction": camera_direction})
    @property
    def objects_found(self):
        return self["objects_found"]
    @property
    def camera_direction(self):
        return self["camera_direction"]
    def copy(self):
        return self.__class__(tuple(self["pose"]), self.camera_direction)


class JointState(oopomdp.OOState):
    def __init__(self, object_states):
        super().__init__(object_states)

    def copy(self):
        object_states = {objid : self.object_states[objid].copy()
                         for objid in self.object_states}
        return JointState(object_states)

    def obj_pose(self, objid):
        return self.object_states[objid]["pose"]

    def __str__(self):
        return "%s(%s)" % (self.__class__.__name__, str(self.object_states))
    


if __name__ == "__main__":
    robot_state = RobotState((10,5), (1,), "+x")
    print(robot_state)
