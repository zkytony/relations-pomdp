import pomdp_py
from corrsearch.objects.object_state import ObjectState

class Move(pomdp_py.SimpleAction):
    """Move action, changes the robot state"""
    def __init__(self, delta, name=None, energy_cost=0, **kwargs):
        if name is None:
            name = str(delta)
        self.name = name
        self.delta = delta
        self.energy_cost = 0
        super().__init__("move-{}".format(self.name))

class UseDetector(pomdp_py.SimpleAction):
    """Use detector action, applies a detector"""
    def __init__(self, detector_id, name=None, energy_cost=0, **kwargs):
        if name is None:
            name = "detector_%d" % detector_id
        self.name = name
        self.detector_id = detector_id
        self.energy_cost = 0
        super().__init__("use-detector-{}-{}".format(detector_id, self.name))

class Declare(pomdp_py.SimpleAction):
    """Declare action. Declare target found"""
    def __init__(self, loc=None, **kwargs):
        self.loc = loc
        self.energy_cost = 0
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
                 actions,
                 trans_model):
        """
        Args:
            object_detectors (array-like): Object detectors,
                each can be thought of as a SensorModel
            actions (array-like): Actions that the robot can perform
            trans_model (TransitionModel): a POMDP transition model
                for the robot state.
        """
        self.detectors = object_detectors
        self.trans_model = trans_model
        self.actions = actions


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
