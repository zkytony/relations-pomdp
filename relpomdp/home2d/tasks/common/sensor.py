"""Sensor model (for example, laser scanner)"""

import math
import numpy as np

# Utility functions
def euclidean_dist(p1, p2):
    return math.sqrt(sum([(a - b)** 2 for a, b in zip(p1, p2)]))

def to_rad(deg):
    return deg * math.pi / 180.0

def in_range(val, rang):
    # Returns True if val is in range (a,b); Inclusive.
    return val >= rang[0] and val <= rang[1]

def in_range_angle(target, angle1, angle2):
    r_angle = ((angle2 - angle1) % 2*math.pi + 2*math.pi) % 2*math.pi;
    if r_angle >= 180:
        angle1, angle2 = angle2, angle1

    if angle1 <= angle2:
        return target >= angle1 and target <= angle2
    else:
        return target >= angle1 or target <= angle2

#### Sensors ####
class Sensor:
    LASER = "laser"
    PROXIMITY = "proximity"
    def observe(self, robot_pose, env_state):
        """
        Returns an Observation with this sensor model.
        """
        raise NotImplementedError

    def within_range(self, robot_pose, point):
        """Returns true if the point is within range of the sensor; but the point might not
        actually be visible due to occlusion or "gap" between beams"""
        raise ValueError

    @property
    def sensing_region_size(self):
        return self._sensing_region_size

    @property
    def robot_id(self):
        # id of the robot equipped with this sensor
        return self._robot_id

class Laser2DSensor:
    """Fan shaped 2D laser sensor
    It is assumed that this sensor will be operating on a single map,
    thus we can maintain a cache to compute more quickly if a point
    is observable or not."""

    def __init__(self, robot_id, grid_map=None,
                 fov=90, min_range=1, max_range=5,
                 angle_increment=5):
        """
        fov (float): angle between the start and end beams of one scan (degree).
        min_range (int or float)
        max_range (int or float)
        angle_increment (float): angular distance between measurements (rad).
        """
        self.robot_id = robot_id
        self.grid_map = grid_map
        self.fov = to_rad(fov)  # convert to radian
        self.min_range = min_range
        self.max_range = max_range
        self.angle_increment = to_rad(angle_increment)
        self._cache = {}  # maps from (robot_pose, point) -> T/F

        # determines the range of angles;
        # For example, the fov=pi, means the range scanner scans 180 degrees
        # in front of the robot. By our angle convention, 180 degrees maps to [0,90] and [270, 360]."""
        self._fov_left = (0, self.fov / 2)
        self._fov_right = (2*math.pi - self.fov/2, 2*math.pi)

        # beams that are actually within the fov (set of angles)
        self._beams = {round(th, 2)
                       for th in np.linspace(self._fov_left[0],
                                             self._fov_left[1],
                                             int(round((self._fov_left[1] - self._fov_left[0]) / self.angle_increment)))}\
                    | {round(th, 2)
                       for th in np.linspace(self._fov_right[0],
                                             self._fov_right[1],
                                             int(round((self._fov_right[1] - self._fov_right[0]) / self.angle_increment)))}
        # The size of the sensing region here is the area covered by the fan
        self._sensing_region_size = self.fov / (2*math.pi) * math.pi * (max_range - min_range)**2


    def in_field_of_view(th, view_angles):
        """Determines if the beame at angle `th` is in a field of view of size `view_angles`.
        For example, the view_angles=180, means the range scanner scans 180 degrees
        in front of the robot. By our angle convention, 180 degrees maps to [0,90] and [270, 360]."""
        fov_right = (0, view_angles / 2)
        fov_left = (2*math.pi - view_angles/2, 2*math.pi)

    def within_range(self, robot_pose, point, grid_map=None, return_intersecting_wall=False):
        """Returns true if the point is within range of the sensor (i.e. visible).
        To do this need to check if any wall is closer to the robot along the
        direction of the beam from robot to the `point`.x

        Args:
            robot_pose (tuple): (x,y,theta) pose of the robot
            point (tuple): (x,y) point to determine if in range
            grid_map (GridMap): The grid map whose walls can cause occlusions
            return_intersecting_wall (bool): True if want to return the wall that
                is intersecting the sensor beam. NOTE: This parameter affects what
                the cache stores. If True, then the cache will also store the intersecting
                wall for this (robot_pose, point) pair."""
        intersecting_wall = None
        if grid_map is None:
            grid_map = self.grid_map
        if (robot_pose, point) in self._cache:
            return self._cache[(robot_pose, point)]
        if robot_pose[:2] == point:
            result = True
        else:
            result = True
            point_dist, point_bearing = self.shoot_beam(robot_pose, point)
            point_in_range = (in_range(point_dist, (self.min_range, self.max_range)))\
                and (in_range(point_bearing, self._fov_left)\
                     or in_range(point_bearing, self._fov_right))
            if not point_in_range:
                result = False
            else:
                for objid in grid_map.walls:
                    wall = grid_map.walls[objid]
                    if wall.intersect(robot_pose[:2], point):
                        result = False
                        intersecting_wall = (objid, wall)
                        break
        if return_intersecting_wall:
            self._cache[(robot_pose, point)] = (result, intersecting_wall)
            return result, intersecting_wall
        else:
            self._cache[(robot_pose, point)] = result
            return result

    def shoot_beam(self, robot_pose, point):
        """Shoots a beam from robot_pose at point. Returns the distance and bearing
        of the beame (i.e. the length and orientation of the beame)"""
        rx, ry, rth = robot_pose
        dist = euclidean_dist(point, (rx,ry))
        bearing = (math.atan2(point[1] - ry, point[0] - rx) - rth) % (2*math.pi)  # bearing (i.e. orientation)
        return (dist, bearing)

    def valid_beam(self, dist, bearing):
        """Returns true beam length (i.e. `dist`) is within range and its angle
        `bearing` is valid, that is, it is within the fov range and in
        accordance with the angle increment."""
        return dist >= self.min_range and dist <= self.max_range\
            and round(bearing, 2) in self._beams

    def _build_beam_map(self, beam, point, beam_map={}):
        """beam_map (dict): Maps from bearing to (dist, point)"""
        dist, bearing = beam
        valid = self.valid_beam(dist, bearing)
        if not valid:
            return
        bearing_key = round(bearing,2)
        if bearing_key in beam_map:
            # There's an object covered by this beame already.
            # see if this beame is closer
            if dist < beam_map[bearing_key][0]:
                # point is closer; Update beam map
                print("HEY")
                beam_map[bearing_key] = (dist, point)
            else:
                # point is farther than current hit
                pass
        else:
            beam_map[bearing_key] = (dist, point)

    @property
    def sensing_region_size(self):
        raise NotImplementedError


class ProximitySensor(Laser2DSensor):
    """This is a simple sensor; Observes a region centered
    at the robot."""
    def __init__(self, robot_id,
                 radius=5,
                 occlusion_enabled=False):
        """
        radius (int or float) radius of the sensing region.
        """
        self.robot_id = robot_id
        self.radius = radius
        self._occlusion_enabled = occlusion_enabled

        # This is in fact just a specific kind of Laser2DSensor
        # that has a 360 field of view, min_range = 0.1 and
        # max_range = radius
        if occlusion_enabled:
            angle_increment = 5
        else:
            angle_increment = 0.25
        super().__init__(robot_id,
                         fov=360,
                         min_range=0.1,
                         max_range=radius,
                         angle_increment=angle_increment,
                         occlusion_enabled=occlusion_enabled)
