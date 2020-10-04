import pomdp_py
from relpomdp.home2d.domain.action import *
from relpomdp.home2d.utils import euclidean_dist
import random

class PolicyModel(pomdp_py.RolloutPolicy):

    """
    Policy model for the primitive actions
    """

    def __init__(self,
                 robot_id, motions={MoveN, MoveS, MoveE, MoveW},
                 other_actions=set(),
                 grid_map=None):
        self.robot_id = robot_id
        self.legal_motions = None
        if grid_map is not None:
            self.legal_motions = grid_map.compute_legal_motions(motions)
        self._motion_actions = motions  # motion actions only
        self._other_actions = other_actions
        self._actions = self._motion_actions | other_actions  # all actions

    def sample(self, state, **kwargs):
        return random.sample(self._get_all_actions(**kwargs), 1)[0]        
    
    def get_all_actions(self, state=None, history=None):
        """
        get_all_actions(self, *args, **kwargs)
        Returns a set of all possible actions, if feasible."""
        if state is None or self.legal_motions is None:
            return self._actions
        else:
            robot_state = state.object_states[self.robot_id]
            motions = self.legal_motions[robot_state["pose"][:2]]
            return motions | self._other_actions

    @property
    def all_motion_actions(self):
        return self._motion_actions

    def rollout(self, state, history=None):
        return random.sample(self.get_all_actions(state=state, history=history), 1)[0]
        

# Preferred policy, action prior.    
class PreferredPolicyModel(PolicyModel):
    """The same with PolicyModel except there is a preferred rollout policypomdp_py.RolloutPolicy"""
    def __init__(self, action_prior, other_actions=set()):
        self.action_prior = action_prior
        super().__init__(self.action_prior.robot_id,
                         self.action_prior.motions,
                         grid_map=self.action_prior.grid_map,
                         other_actions=other_actions)
        
    def rollout(self, state, history):
        # Obtain preference and returns the action in it.
        preferences = self.action_prior.get_preferred_actions(state, history)
        if len(preferences) > 0:
            return random.sample(preferences, 1)[0][0]
        else:
            return random.sample(self.get_all_actions(state=state, history=history), 1)[0]
