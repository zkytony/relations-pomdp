import pomdp_py
from corrsearch.objects.object_state import ObjectState

class Move(pomdp_py.SimpleAction):
    """Move action, changes the robot state"""
    def __init__(self, delta, **kwargs):
        self.delta = delta
        super().__init__("move-{}".format(delta))

class UseDetector(pomdp_py.SimpleAction):
    """Use detector action, applies a detector"""
    def __init__(self, detector_id, **kwargs):
        self.detector_id = detector_id
        super().__init__("use-detector-{}".format(detector_id))

class Declare(pomdp_py.SimpleAction):
    """Declare action. Declare target found"""
    def __init__(self, loc=None, **kwargs):
        self.loc = loc
        if loc is None:
            super().__init__("declare")
        else:
            super().__init__("declare-{}".format(loc))

class RobotModel:
    """
    A robot model has:
    - a set of object detectors
        each represented as a sensor observation model
    - a model of the robot state transition,
        (actions are MOVE, LOOK, DECLARE)
    """
    def __init__(self,
                 object_detectors,
                 trans_model):
        """
        Args:
            object_detectors (array-like): Object detectors,
                each can be thought of as a SensorModel
            trans_model (TransitionModel): a POMDP transition model
                for the robot state.
        """
        self.detectors = object_detectors
        self.trans_model = trans_model


class RobotTransModel(pomdp_py.TransitionModel):
    """The transition model is factored"""
    def probability(self, next_robot_state, state, action, **kwargs):
        """
        Pr(s_r' | s, a)
        """
        raise NotImplementedError

    def sample(self, state, action, **kwargs):
        """
        s_r' ~ T(s,a)
        """
        raise NotImplementedError
