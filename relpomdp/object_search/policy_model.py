import pomdp_py
from relpomdp.object_search.action import *
import random

class PolicyModel(pomdp_py.RolloutPolicy):

    def __init__(self, ids, motion_policy):
        self.ids = ids
        self.motion_policy = motion_policy
        
    def sample(self, state, **kwargs):
        return random.sample(self._get_all_actions(**kwargs), 1)[0]        
    
    def get_all_actions(self, state=None, history=None):
        """
        get_all_actions(self, *args, **kwargs)
        Returns a set of all possible actions, if feasible."""
        if state is None:
            return {MoveE, MoveW, MoveN, MoveS, Pickup()}
        else:
            robot_state = state.object_states[self.ids["Robot"]]
            motions = self.motion_policy.valid_motions(robot_state.pose)
            return motions | {Pickup()}

    def rollout(self, state, history=None):
        return random.sample(self.get_all_actions(state=state, history=history), 1)[0]
        
