# from moos3d.environment.env import Mos3D
# from pomdp_py import OOPOMDP, Environment
from pomdp_py import ObjectState as ppObjectState
from pomdp_py import OOState as ppOOState
import random

class ObjectState(ppObjectState):
    def __init__(self, objid, objclass, pose, res=1):
        super().__init__(objclass, {"pose":pose, "id":objid, "res":res})
    def __str__(self):
        return '%s%s' % (str(self.objclass), str(self.pose))
    @property
    def pose(self):
        return self.attributes['pose']
    @property
    def res(self):
        return self.attributes['res']
    @property
    def resolution(self):
        return self.res
    @property
    def objid(self):
        return self.attributes['id']

class RobotState(ObjectState):
    def __init__(self, robot_id, pose, objects_found, camera_direction, res=1):
        """Note: camera_direction is None unless the robot is looking at a direction,
        in which case camera_direction is the string e.g. look+thz, or 'look'"""
        super().__init__("robot", {"id":robot_id,
                                   "pose":pose,
                                   "res": res,
                                   "objects_found": objects_found,
                                   "camera_direction": camera_direction})
    def __str__(self):
        return 'RobotState(%s%s|%s)' % (str(self.objclass), str(self.pose), str(self.objects_found))
    def __repr__(self):
        return str(self)
    @property
    def pose(self):
        return self.attributes['pose']
    @property
    def robot_pose(self):
        return self.attributes['pose']
    @property
    def objects_found(self):
        return self.attributes['objects_found']
    @property
    def res(self):
        return self.attributes['res']
    @property
    def resolution(self):
        return self.res    

class JointState(ppOOState):
    def __init__(self, robot_id, object_states):
        self._robot_id = robot_id
        super().__init__(object_states)
    def get_robot_state(self):
        return self.object_states[self._robot_id]
    @property
    def robot_id(self):
        return self._robot_id
    @property
    def robot_pose(self):
        return self.object_states[self._robot_id]['pose']
    @property
    def object_poses(self):
        return {objid:self.object_states[objid]['pose']
                for objid in self.object_states
                if objid != self._robot_id}
    @property
    def robot_state(self):
        return self.object_states[self._robot_id]
    def __str__(self):
        return 'JointState(%d)%s' % (self._robot_id, str(self.object_states))
    def __repr__(self):
        return str(self)    

