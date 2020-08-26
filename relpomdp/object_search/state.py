import numpy as np
import pomdp_py
import relpomdp.oopomdp.framework as oopomdp
    
# Object classes and attributes
class PoseState(oopomdp.ObjectState):
    @property
    def pose(self):
        return self["pose"]
    
class ContainerState(oopomdp.ObjectState):
    @property
    def footprint(self):
        return self["footprint"]

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

    
class RoomState(ContainerState):
    def __init__(self, footprint, room_type="Room"):
        super().__init__(room_type,
                         {"footprint": footprint})
    @property
    def room_type(self):
        return self.objclass
    def copy(self):
        return self.__class__(tuple(self.footprint), room_type=self.room_type)
    
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


if __name__ == "__main__":
    robot_state = RobotState((10,5), (1,), "+x")
    print(robot_state)
