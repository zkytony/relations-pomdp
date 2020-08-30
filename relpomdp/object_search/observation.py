import numpy as np
import pomdp_py
import relpomdp.oopomdp.framework as oopomdp

# Object classes and attributes
class PoseObservation(oopomdp.ObjectObservation):
    @property
    def pose(self):
        return self["pose"]
    
class ItemObservation(PoseObservation):
    def __init__(self, name, pose):
        super().__init__(name,
                         {"pose": pose})
    @property
    def name(self):
        return self.objclass
    def copy(self):
        return self.__class__(self.name, tuple(self["pose"]))

    def to_evidence(self):
        return {"%s_pose" % (self.name): self["pose"]}

class JointObservation(oopomdp.OOObservation):
    def __init__(self, object_observations):
        super().__init__(object_observations)

    def copy(self):
        object_observations = {objid : self.object_observations[objid].copy()
                               for objid in self.object_observations}
        return JointObservation(object_observations)

    def obj_pose(self, objid):
        return self.object_observations[objid]["pose"]

    def for_objs(self, objids):
        object_observations = {objid : self.object_observations[objid].copy()
                               for objid in objids\
                               if objid in self.object_observations}
        return JointObservation(object_observations)


if __name__ == "__main__":
    robot_observation = RobotObservation((10,5), (1,), "+x")
    print(robot_observation)
