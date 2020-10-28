import pomdp_py
from relpomdp.home2d.domain.condition_effect import MoveEffect
from relpomdp.home2d.domain.action import *
from relpomdp.home2d.utils import euclidean_dist
import random

def split_actions(actions):
    """Split actions into move and non-move actions"""
    motions = {a for a in actions
                     if isinstance(a, Move)}
    other_actions = actions - motions
    return motions, other_actions

class RandomPolicyModel(pomdp_py.RolloutPolicy):

    """
    Policy model for the primitive actions
    """

    def __init__(self,
                 robot_id,
                 actions,  # a set of valid actions
                 legal_motions=None,
                 memory={}):
        self.robot_id = robot_id
        self.legal_motions = legal_motions
        self._actions = actions
        self._motions, self._other_actions = split_actions(actions)
        self._memory = memory  # map from robot_pose to actions

    @property
    def memory(self):
        """Remembers which actions were allowed at each robot pose"""
        return self._memory

    def sample(self, state, **kwargs):
        return random.sample(self._get_all_actions(**kwargs), 1)[0]

    def get_all_actions(self, state=None, history=None):
        """
        get_all_actions(self, *args, **kwargs)
        Returns a set of all possible actions, if feasible."""
        robot_state = state.object_states[self.robot_id]
        robot_pose = robot_state["pose"]
        if robot_pose in self._memory:
            return self._memory[robot_pose]
        else:
            motions = self.legal_motions[robot_pose[:2]]
            return motions | self._other_actions

    @property
    def all_motion_actions(self):
        return self._motion_actions

    def rollout(self, state, history=None):
        return random.sample(self.get_all_actions(state=state, history=history), 1)[0]

    def update(self, robot_pose, next_robot_pose, action, **kwargs):
        """Record invalid move action which did not result in the robot moving"""
        if isinstance(action, Move) and next_robot_pose == robot_pose:
            self._record_invalid_action(robot_pose, action)

    def _record_invalid_action(self, robot_pose, action):
        if robot_pose not in self._memory:
            motions = self.legal_motions[robot_pose[:2]]
            self._memory[robot_pose] = (motions - set({action})) | self._other_actions
        else:
            self._memory[robot_pose] -= set({action})


# Preferred policy, action prior.
class PreferredPolicyModel(RandomPolicyModel):
    """The same with PolicyModel except there is a preferred rollout policypomdp_py.RolloutPolicy"""
    def rollout(self, state, history):
        # Obtain preference and returns the action in it.
        if not hasattr(self, "action_prior"):
            raise ValueError("PreferredPolicyModel is not assigned an action prior.")
        preferences = self.action_prior.get_preferred_actions(state, history)
        if len(preferences) > 0:
            return random.sample(preferences, 1)[0][0]
        else:
            return random.sample(self.get_all_actions(state=state, history=history), 1)[0]

    def add_action_prior(self, action_prior_class, *args, **kwargs):
        self.action_prior = action_prior_class.build(self, *args, **kwargs)


class GreedyActionPrior(pomdp_py.ActionPrior):
    """This greedy policy model prefers actions that
    moves the robot towards a given object with given id"""

    @classmethod
    def build(cls, policy_model, objid, num_visits_init=10, val_init=100):
        return GreedyActionPrior(objid, policy_model,
                                 num_visits_init=num_visits_init,
                                 val_init=val_init)

    def get_preferred_actions(self, state, history):
        robot_state = state.object_states[self._p.robot_id]
        obj_state = state.object_states[self.objid]
        # If applying an action moves the robot on top of the object,
        # then we prefer this motion.
        preferences = set()
        for action in self._p.legal_motions[robot_state["pose"][:2]]:
            rx, ry, th = MoveEffect.move_by(robot_state["pose"][:2], action)
            if (rx, ry) == obj_state["pose"]:
                preferences.add((action, self.num_visits_init, self.val_init))
        return preferences

    def __init__(self,
                 objid,
                 policy_model,
                 num_visits_init=1,
                 val_init=100):
        """This greedy policy model prefers actions that
        moves the robot on top of a given object with given id

        Should never be called by the user;
        The user should always only use a PreferredPolicyModel,
        and the `add_prior` function which will construct a prior,
        that calls this function appropriately."""
        self._p = policy_model
        self.num_visits_init = num_visits_init
        self.val_init = val_init

        # The given object
        self.objid = objid
