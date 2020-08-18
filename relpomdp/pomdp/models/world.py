from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
from OpenGL.arrays import vbo
import random
import numpy as np

from relpomdp.models.objects import *
from relpomdp.models.sensor import FrustumCamera
import moos3d.util as util

OBJECT_MANAGER = GObjManager()
OBJECT_MANAGER.register_all([(Robot, 1, 'robot'),
                             (Cube, 2, 'cube'),
                             (OrangeRicky, 3, 'orange_ricky'),
                             (Hero, 4, 'hero'),
                             (Teewee, 5, 'teewee'),
                             (Smashboy, 6, 'smashboy')])            

def diff(rang):
    return rang[1] - rang[0]

class GridWorld:

    """Gridworld provides implementation of motion and observation functions
    but does not store the true state. The true state is only stored by
    the Environment. The Gridworld does require general information such
    as objects (their ids, types) and world boundary."""

    def __init__(self, w, l, h, robot, objects,
                 robot_id=0, occlusion_enabled=False,
                 obstacles=set({}), hidden=set({})):
        """
        robot (Robot); no pose or other variable information is stored
        objects (dict) {id -> obj(Object); no pose is stored }.
        obstacles (set): set of object ids (subset of objects.keys()) that
                         represent objects which are obstacles.
        hidden (set): set of grid locations (x,y,z) that will always be UNKNOWN,
                      as if it is occluded.
        """
        self._w, self._l, self._h = w, l, h
        self._robot_id = robot_id
        self._robot = robot
        self._objects = objects
        self._obstacles = obstacles
        self._hidden = hidden
        self._target_objects = set(objects.keys()) - obstacles
        self._x_range = (0, self._w-1)
        self._y_range = (0, self._l-1)
        self._z_range = (0, self._h-1)
        self._occlusion_enabled = occlusion_enabled
        self._observation_cache = {}  # maps from (robot_pose, object_poses) to {set of observable objects}

    def valid_pose(self, pose, object_poses=None, check_collision=True):
        x, y, z = pose[:3]

        # Check collision
        if check_collision and object_poses is not None:
            for objid in object_poses:
                true_object_pose = object_poses[objid]
                for cube_pose in self._objects[objid].cube_poses(*true_object_pose):
                    if (x,y,z) == tuple(cube_pose):
                        return False
        return self.in_boundary(pose)

    def is_obstacle(self, objid):
        return objid in self._obstacles

    @property
    def target_objects(self):
        # Returns a set of objids for target objects
        return self._target_objects
        
    def in_boundary(self, pose):
        # Check if in boundary
        x,y,z = pose[:3]
        if x >= 0 and x < self.width:
            if y >= 0 and y < self.length:
                if z >= 0 and z < self.height:
                    if len(pose) > 3 and len(pose) < 7:
                        # check if orientation is valid
                        thx, thy, thz = pose[3:]
                        if thx >= 0 and thx <= 360:
                            if thy >= 0 and thy <= 360:
                                if thz >= 0 and thz <= 360:
                                    return True
                    elif len(pose) == 7:
                        # check if quaternion is valid (unorm=1)
                        qx, qy, qz, qw = pose[3:]
                        return abs((np.linalg.norm([qx,qy,qz,qw])) - 1.0) <= 1e-6
                    else:
                        return True
        return False        

    # MOTION MODEL
    def if_move_by(self, *params, motion_model="AXIS", valid_pose_func=None, object_poses=None,
                   absolute_rotation=False):
        if motion_model == "AXIS":
            return self.if_move_by_axis(*params, valid_pose_func, object_poses=object_poses, absolute_rotation=absolute_rotation)
        elif motion_model == "FORWARD":
            raise ValueError("FORWARD Motion Model is deprecated")
            return self.if_move_by_forward(*params, valid_pose_func)
        else:
            raise ValueError("Unknown motion model %s" % motion_model)

    def if_move_by_axis(self, cur_pose, dpos, dth, valid_pose_func, object_poses=None,
                        absolute_rotation=False):
        """Returns the pose the robot if the robot is moved by the given control.
        The robot is not actually moved underneath;
        There's no check for collision now."""
        robot_pose = [0, 0, 0, 0, 0, 0, 0]
        
        robot_pose[0] = cur_pose[0] + dpos[0]#max(0, min(, self.width-1)) 
        robot_pose[1] = cur_pose[1] + dpos[1]#max(0, min(, self.length-1))
        robot_pose[2] = cur_pose[2] + dpos[2]#max(0, min(, self.height-1))

        # Use quaternion
        if not absolute_rotation:
            qx, qy, qz, qw = cur_pose[3:]
        else:
            qx, qy, qz, qw = 0, 0, 0, 1
        R = util.R_quat(qx, qy, qz, qw)
        if dth[0] != 0 or dth[1] != 0 or dth[2] != 0:
            R_prev = R
            R_change = util.R_quat(*util.euler_to_quat(dth[0], dth[1], dth[2]))
            R = R_change * R_prev
        robot_pose[3], robot_pose[4], robot_pose[5], robot_pose[6] = R.as_quat()

        if valid_pose_func is not None and valid_pose_func(robot_pose, object_poses=object_poses):
            return tuple(robot_pose)
        else:
            return cur_pose

    # Deprecated! (fix in rotation not applied)
    def if_move_by_forward(self, cur_pose, forward, dth, valid_pose_func):
        robot_facing = self.get_camera_direction(cur_pose, get_tuple=False)
        # project this vector to xy plane, then obtain the "shadow" on xy plane
        forward_vec = robot_facing*forward
        xy_shadow = forward_vec - util.proj(forward_vec, np.array([0,0,1]))
        dy = util.proj(xy_shadow[:2], np.array([0,1]), scalar=True)
        dx = util.proj(xy_shadow[:2], np.array([1,0]), scalar=True)
        yz_shadow = forward_vec - util.proj(forward_vec, np.array([1,0,0]))
        dz = util.proj(yz_shadow[1:], np.array([0,1]), scalar=True)

        dpos = (dx, dy, dz)
        robot_pose = np.array([0,0,0,0,0,0])
        robot_pose[0] = max(0, min(cur_pose[0] + round(dpos[0]), self.width-1)) 
        robot_pose[1] = max(0, min(cur_pose[1] + round(dpos[1]), self.length-1))
        robot_pose[2] = max(0, min(cur_pose[2] + round(dpos[2]), self.height-1))
        robot_pose[3] = (cur_pose[3] + dth[0]) % 360                     
        robot_pose[4] = (cur_pose[4] + dth[1]) % 360                     
        robot_pose[5] = (cur_pose[5] + dth[2]) % 360
        if valid_pose_func(robot_pose):
            return tuple(robot_pose)
        else:
            return cur_pose
    
    @property
    def width(self):
        return diff(self._x_range)
    @property
    def length(self):
        return diff(self._y_range)
    @property
    def height(self):
        return diff(self._z_range)
    @property
    def objects(self):
        return self._objects
    @property
    def robot(self):
        return self._robot
    @property
    def robot_id(self):
        return self._robot_id

