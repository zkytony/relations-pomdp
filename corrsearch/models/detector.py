import pomdp_py
from corrsearch.objects import JointObz, ObjectObz
from corrsearch.models.robot_model import UseDetector
from corrsearch.utils import *

class NullObz(ObjectObz):
    """Didn't observe the object"""
    def __init__(self, objid):
        super().__init__(objid, None, {})

class DetectorModel(pomdp_py.ObservationModel):
    """Detector Model is the sensor model of the detector.
    The observation should be factored by objects,
    and for each object, we have Pr(zi | si, sr', a).
    This model should have a way to sample and evaluate
    this per-object model

    Convention: An empty observation for an object may be received.  There is no
    notion of "joint null observation". The observation is always factored, and
    could be factored into null observation for every object.
    """
    def __init__(self, detector_id, robot_id):
        self.id = detector_id
        self.robot_id = robot_id

    def iprob(self, objobz, objstate, robot_state, action):
        """
        Returns the probability of Pr(zi | si, sr', a)
        """
        raise NotImplementedError

    def isample(self, objstate, robot_state, action):
        """
        Returns a sample zi according to Pr(zi | si, sr', a)
        """
        raise NotImplementedError

    def probability(self, observation, next_state, action, **kwargs):
        """
        Pr(z | s', a)

        Args:
            observation (JointObz): Joint observation
            next_state (JointState): Joint state
            action (Action): action
        """
        if not isinstance(action, UseDetector)\
           or action.detector_id != self.id:
            # .sample should return the joint of null observations, which is expected
            return indicator(self.sample(next_state, action) == observation)

        p = 1.0
        for objid in observation:
            if objid == self.robot_id:
                continue
            zi = observation[objid]
            si = next_state[objid]
            sr = next_state[self.robot_id]
            p *= self.iprob(zi, si, sr, action)
        return p

    def sample(self, next_state, action, **kwargs):
        """
        z ~ O(s',a)

        Args:
            next_state (JointState): Joint state
            action (Action)
        """
        if not isinstance(action, UseDetector)\
           or action.detector_id != self.id:
            return JointObz({objid:NullObz(objid)
                             for objid in next_state
                             if objid != self.robot_id})

        objzs = {}
        for objid in next_state:
            if objid == self.robot_id:
                continue
            si = next_state[objid]
            sr = next_state[self.robot_id]
            objzs[objid] = self.isample(si, sr, action)
        return JointObz(objzs)
