import pomdp_py
import corrsearch.objects
from corrsearch.models.robot_model import Declare

class SearchProblem:
    """
    A search problem. The problem is defined by
    a set of possible locations,
    a set of objects,
    a joint distribution of their locations,
    a target object.
    and a robot model:
    - a set of object detectors,
        each represented as a sensor observation model
    - a model of the robot state transition,
       (actions are MOVE, LOOK, DECLARE)
    """
    def __init__(self,
                 locations,
                 objects,
                 joint_dist,
                 target_object,
                 robot_model):
        self.locations      = locations
        self._objects       = objects
        self.joint_dist     = joint_dist
        self.target_object  = target_object
        self.robot_model    = robot_model

    def instantiate(self, *args, **kwargs):
        """This should return a search problem instance by creating a
        pomdp_py.Environment and an pomdp_py.Agent for it. We CHOOSE not to
        instantiate a custom Environment, but to just create one. Because an
        Environment always needs to have a true state, and an Agent needs to
        have a belief, but a search problem does not.
        """
        raise NotImplementedError

    def visualizer(self, *args, **kwargs):
        """Returns a visualizer of the problem"""
        raise NotImplementedError


class SearchRewardModel(pomdp_py.RewardModel):
    def __init__(self, robot_id, target_id, rmax=100, rmin=-100):
        self.rmax = rmax
        self.rmin = rmin
        self.robot_id = robot_id
        self.target_id = target_id

    def sample(self, state, action, next_state):
        if state[self.robot_id].terminal:
            return 0

        if isinstance(action, Declare):
            if action.loc is None:
                decloc = state[self.robot_id].loc
            else:
                decloc = action.loc
            if decloc == state[self.target_id].loc:
                return self.rmax
            else:
                return self.rmin
        else:
            return self.step_reward_func(state, action, next_state)

    def step_reward_func(self, state, action, next_state):
        raise NotImplementedError
