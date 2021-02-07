import pomdp_py
import random
import numpy as np
from corrsearch.objects.object_obz import ObjectObz
from corrsearch.utils import euclidean_dist, to_rad
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
                 detection_type, **params):
        """
        Args:
            detection_type (str): "loc" or "label".
                If loc, then the non-null observation will
                contain the location of detected object.
                If label, then the non-null observation will
                contain only the detected object label.
            params (dict):
                Parameters of the detector.
        """
        self.detection_type = detection_type
        self.params = params
        super().__init__(detector_id, robot_id)

    def sensor_region(self, objid, robot_state):
        raise NotImplementedError

    def in_range(self, objstate, robot_state):
        raise NotImplementedError

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
                                             [[self.params["sigma"]**2, 0],
                                              [0, self.params["sigma"]**2]])
                return self.params[objobz.objid]["true_positive"] * gaussian[objobz["loc"]]
        else:
            if isinstance(objobz, NullObz):
                # True negative
                return 1.0 - self.params["false_positive"][objobz.objid]
            else:
                return self.params["false_positive"][objobz.objid] * (1 / len(self.sensor_region(objstate.id, robot_state)))

    def _isample_loc(self, objstate, robot_state, in_range):
        if in_range:
            if random.uniform(0,1) <= self.params["true_positive"][objstate.objid]:
                # sample according to gaussian
                gaussian = pomdp_py.Gaussian(list(objstate["loc"]),
                                             [[self.params["sigma"]**2, 0],
                                              [0, self.params["sigma"]**2]])
                loc = tuple(map(int, map(round, gaussian.random())))
                return LocObz(objstate.id, objstate.objclass, loc)
            else:
                return NullObz(objstate.id)
        else:
            if random.uniform(0,1) <= self.params["false_positive"][objstate.objid]:
                # False positive. Can come from anywhere within the sensor
                # Requires to know the detection locations.
                region = self.sensor_region(objstate.id, robot_state)
                return LocObz(objstate.id, objstate.objclass,
                              random.sample(region, 1)[0])
            else:
                # True negative
                return NullObz(objstate.id)

    def iprob(self, objobz, objstate, robot_state, action):
        assert isinstance(action, UseDetector)
        assert action.detector_id == self.id
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
        in_range = self.in_range(objstate, robot_state)
        if self.detection_type == "label":
            return self._isample_label(objstate, in_range)
        elif self.detection_type == "loc":
            return self._isample_loc(objstate, robot_state, in_range)
        else:
            raise ValueError("Cannot handle detection type %s" % self.detection_type)


# Fan shape laser scanner
class LaserRangeDetector(RangeDetector):

    def __init__(self, name="laser2d",
                 fov=90, min_range=1, max_range=5,
                 angle_increment=5):
        self.name = name
        self.robot_id = robot_id
        self.fov = to_rad(fov)  # convert to radian
        self.min_range = min_range
        self.max_range = max_range
        self.angle_increment = to_rad(angle_increment)

    def in_range(self, objstate, robot_state):
        raise NotImplementedError
