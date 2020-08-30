import pomdp_py
from relpomdp.object_search.action import *
from relpomdp.object_search.utils import euclidean_dist
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

    @property
    def all_motion_actions(self):
        return {MoveE, MoveW, MoveN, MoveS}

    def rollout(self, state, history=None):
        return random.sample(self.get_all_actions(state=state, history=history), 1)[0]
        

# Preferred policy, action prior.    
class PreferredPolicyModel(PolicyModel):
    """The same with PolicyModel except there is a preferred rollout policypomdp_py.RolloutPolicy"""
    def __init__(self, action_prior):
        self.action_prior = action_prior
        super().__init__(self.action_prior.ids,
                         self.action_prior.motion_policy)
        
    def rollout(self, state, history):
        # Obtain preference and returns the action in it.
        preferences = self.action_prior.get_preferred_actions(state, history)
        if len(preferences) > 0:
            return random.sample(preferences, 1)[0][0]
        else:
            return random.sample(self.get_all_actions(state=state, history=history), 1)[0]

    
class GreedyActionPrior(pomdp_py.ActionPrior):
    """greedy action prior for 'xy' motion scheme"""
    def __init__(self, ids, motion_policy, num_visits_init, val_init,
                 no_look=False):
        self.ids = ids
        self.motion_policy = motion_policy
        self.all_motion_actions = None
        self.num_visits_init = num_visits_init
        self.val_init = val_init
        self.no_look = no_look
        
    def get_preferred_actions(self, state, history):
        """Get preferred actions. This can be used by a rollout policy as well."""
        robot_id = self.ids["Robot"]
        robot_state = state.object_states[robot_id]
        cur_room = self.motion_policy._grid_map.room_of(robot_state.pose[:2])
        preferences = set()
        for objid in state.object_states:
            objstate = state.object_states[objid]
            if objid in self.ids["Target"] and objstate.is_found is False:
                cur_dist = euclidean_dist(robot_state.pose, objstate.pose)
                neighbors = self.motion_policy.get_neighbors(robot_state.pose)
                for next_robot_pose in neighbors:
                    # # Prefer action to move into a different room
                    action = neighbors[next_robot_pose]
                    next_room = self.motion_policy._grid_map.room_of(next_robot_pose[:2])
                    object_room = self.motion_policy._grid_map.room_of(objstate.pose)
                    if object_room == next_room\
                       and euclidean_dist(next_robot_pose, objstate.pose) < cur_dist:
                        preferences.add((action,
                                         self.num_visits_init, self.val_init))
        return preferences


# Greedy planner
