import pomdp_py
import random
import numpy as np
from corrsearch.objects.object_obz import ObjectObz
from corrsearch.utils import *
from corrsearch.models import *

class LocObz(ObjectObz):
    def __init__(self, objid, objclass, loc):
        super().__init__(objid, objclass, {"loc": loc})

class LabelObz(ObjectObz):
    """This means a positive label detection"""
    def __init__(self, objid, objclass):
        super().__init__(objid, objclass, {"label": objid})

class RangeDetector(DetectorModel):
    def __init__(self, detector_id, robot_id,
                 detection_type, sensors, name=None, energy_cost=0, **params):
        """
        Args:
            detection_type (str): "loc" or "label".
                If loc, then the non-null observation will
                contain the location of detected object.
                If label, then the non-null observation will
                contain only the detected object label.
            params (dict):
                Parameters of the detector.
            sensors (dict): Mapping from object id to sensor.
                Using the given sensor for the object. One sensor
                per object per detector
        """
        self.detection_type = detection_type
        self.params = params
        self.sensors = sensors
        self.energy_cost = energy_cost
        super().__init__(detector_id, robot_id, name=name)

    def sensor_region_size(self, objid, robot_state):
        return self.sensors[objid].sensor_region_size

    def uniform_sample_sensor_region(self, objid, robot_state):
        return self.sensors[objid].uniform_sample_sensor_region(robot_state.pose)

    def in_range(self, objstate, robot_state):
        return self.sensors[objstate.objid].in_range(objstate["loc"], robot_state.pose)

    def _iprob_label(self, objobz, in_range):
        assert isinstance(objobz, LabelObz) or isinstance(objobz, NullObz)
        if in_range:
            if isinstance(objobz, LabelObz):
                return self.params["true_positive"][objobz.objid]
            else:
                return 1.0 - self.params["true_positive"][objobz.objid]
        else:
            if isinstance(objobz, LabelObz):
                return self.params["false_positive"][objobz.objid]
            else:
                return 1.0 - self.params["false_positive"][objobz.objid]

    def _isample_label(self, objstate, in_range):
        if in_range:
            if random.uniform(0,1) <= self.params["true_positive"][objstate.objid]:
                return LabelObz(objstate.id, objstate.objclass)
            else:
                return NullObz(objstate.id)
        else:
            if random.uniform(0,1) <= self.params["false_positive"][objstate.objid]:
                return LabelObz(objstate.id, objstate.objclass)
            else:
                return NullObz(objstate.id)


    def _iprob_loc(self, objobz, objstate, robot_state, in_range):
        # """This is a similar observation model as in the OOPOMDP paper"""
        assert isinstance(objobz, LocObz) or isinstance(objobz, NullObz)
        if in_range:
            if isinstance(objobz, NullObz):
                # False negative
                return 1.0 - self.params["true_positive"][objobz.objid]
            else:
                # True positive, gaussian centered at robot pose
                gaussian = pomdp_py.Gaussian(list(objstate["loc"]),
                                             [[self.params["sigma"][objstate.objid]**2, 0],
                                              [0, self.params["sigma"][objstate.objid]**2]])
                return self.params["true_positive"][objobz.objid] * gaussian[objobz["loc"]]
        else:
            if isinstance(objobz, NullObz):
                # True negative
                return 1.0 - self.params["false_positive"][objobz.objid]
            else:
                return self.params["false_positive"][objobz.objid] * (1 / self.sensor_region_size(objstate.id, robot_state))

    def _isample_loc(self, objstate, robot_state, in_range):
        if in_range:
            if random.uniform(0,1) <= self.params["true_positive"][objstate.objid]:
                # sample according to gaussian
                gaussian = pomdp_py.Gaussian(list(objstate["loc"]),
                                             [[self.params["sigma"][objstate.objid]**2, 0],
                                              [0, self.params["sigma"][objstate.objid]**2]])
                loc = tuple(map(int, map(round, gaussian.random())))
                return LocObz(objstate.id, objstate.objclass, loc)
            else:
                return NullObz(objstate.id)
        else:
            if random.uniform(0,1) <= self.params["false_positive"][objstate.objid]:
                # False positive. Can come from anywhere within the sensor
                # Requires to know the detection locations.
                return LocObz(objstate.id, objstate.objclass,
                              self.uniform_sample_sensor_region(objstate.id, robot_state))
            else:
                # True negative
                return NullObz(objstate.id)

    def iprob(self, objobz, objstate, robot_state, action):
        assert isinstance(action, UseDetector)
        assert action.detector_id == self.id
        if objobz.id not in self.sensors:
            # Not capable of detecting the given object
            return indicator(objobz == NullObz(objobz.id))

        in_range = self.in_range(objstate, robot_state)
        if self.detection_type == "label":
            return self._iprob_label(objobz, in_range)
        elif self.detection_type == "loc":
            return self._iprob_loc(objobz, objstate, robot_state, in_range)
        else:
            raise ValueError("Cannot handle detection type %s" % self.detection_type)


    def isample(self, objstate, robot_state, action):
        assert isinstance(action, UseDetector)
        assert action.detector_id == self.id
        if objstate.id not in self.sensors:
            # Not capable of detecting the given object
            return NullObz(objstate.id)

        in_range = self.in_range(objstate, robot_state)
        if self.detection_type == "label":
            return self._isample_label(objstate, in_range)
        elif self.detection_type == "loc":
            return self._isample_loc(objstate, robot_state, in_range)
        else:
            raise ValueError("Cannot handle detection type %s" % self.detection_type)


class Sensor:
    """Sensor. Not tied to any particular object"""
    def in_range(self, robot_pose, object_pose):
        raise NotImplementedError
    def sensor_region(self, robot_pose):
        raise NotImplementedError
    @property
    def sensor_region_size(self):
        raise NotImplementedError
    def uniform_sample_sensor_region(self, robot_pose):
        """Returns a location in the field of view
        uniformly at random"""
        raise NotImplementedError


class FanSensor(Sensor):
    def __init__(self, name="laser2d_sensor", **params):
        fov = params.get("fov", 90)
        min_range = params.get("min_range", 1)
        max_range = params.get("max_range", 5)
        self.name = name
        self.fov = to_rad(fov)  # convert to radian
        self.min_range = min_range
        self.max_range = max_range
        # The size of the sensing region here is the area covered by the fan
        # This is a float, but rounding it up should equal to the number of discrete locations
        # in the field of view.
        self._sensing_region_size = int(math.ceil(self.fov / (2*math.pi) * math.pi * (max_range - min_range)**2))

    def uniform_sample_sensor_region(self, robot_pose):
        """Returns a location in the field of view
        uniformly at random. Expecting robot pose to
        have x, y, th, where th is in radians."""
        assert len(robot_pose) == 3,\
            "Robot pose must have x, y, th"
        # Sample a location (r,th) for the default robot pose
        th = random.uniform(0, self.fov) - self.fov/2
        r = random.uniform(self.min_range, self.max_range+1)
        x, y = pol2cart(r, th)
        # transform to robot pose
        x, y = np.matmul(R2d(robot_pose[2]), np.array([x,y])) # rotation
        x += robot_pose[0]  # translation dx
        y += robot_pose[1]  # translation dy
        point = int(x), int(y)
        if not self.in_range(point, robot_pose):
            return self.uniform_sample_sensor_region(robot_pose)
        else:
            return point

    @property
    def sensor_region_size(self):
        return self._sensing_region_size

    def in_range(self, point, robot_pose):
        if robot_pose[0] == point and self.min_range == 0:
            return True

        dist, bearing = self.shoot_beam(robot_pose, point)
        if self.min_range <= dist <= self.max_range:
            # because we defined bearing to be within 0 to 360, the fov
            # angles should also be defined within the same range.
            fov_ranges = (0, self.fov/2), (2*math.pi - self.fov/2, 2*math.pi)
            if in_range_inclusive(bearing, fov_ranges[0])\
               or in_range_inclusive(bearing, fov_ranges[1]):
                return True
            else:
                return False
        return False

    def shoot_beam(self, robot_pose, point):
        """Shoots a beam from robot_pose at point. Returns the distance and bearing
        of the beame (i.e. the length and orientation of the beame)"""
        rx, ry, rth = robot_pose
        dist = euclidean_dist(point, (rx,ry))
        bearing = (math.atan2(point[1] - ry, point[0] - rx) - rth) % (2*math.pi)  # bearing (i.e. orientation)
        return (dist, bearing)


class DiskSensor(Sensor):
    """Field of view is a disk"""
    def __init__(self, name="disk_sensor", **params):
        self.name = name
        self.radius = params.get("radius", 5)
        self._sensor_region = np.array(
            [
                [x,y]
                for x in range(-2*self.radius-1, 2*self.radius+1)
                for y in range(-2*self.radius-1, 2*self.radius+1)
                if x**2 + y**2 <= self.radius**2
            ]
        )
        self._sensor_region_set = set(map(tuple, self._sensor_region.tolist()))

    def in_range(self, point, robot_pose):
        return euclidean_dist(robot_pose[:2],
                              point) <= self.radius

    def sensor_region(self, robot_pose):
        region = self._sensor_region + np.array(robot_pose[:2])
        return region

    @property
    def sensor_region_size(self):
        return len(self._sensor_region)

    def uniform_sample_sensor_region(self, robot_pose):
        point = random.sample(self._sensor_region_set, 1)[0]
        return (point[0] + robot_pose[0],
                point[1] + robot_pose[1])
