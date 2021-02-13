import pomdp_py
import random
from corrsearch.objects import *
from corrsearch.models.robot_model import *

class SearchTransitionModel(pomdp_py.TransitionModel):
    def __init__(self, robot_id, robot_trans_model, target_states=None):
        """
        To enable explicit enumeration of the state space, pass in locations,
        and objects. The state space will be the set of joint configuration
        of the robot and the target states.

        Args:
            robot_id (int)
            robot_trans_model (RobotTransModel)
        """
        self.robot_id = robot_id
        self.robot_trans_model = robot_trans_model
        self.target_states = target_states

    def probability(self, next_state, state, action, **kwargs):
        prob = 1.0
        for objid in next_state:
            if objid == self.robot_id:
                robot_state = next_state[objid]
                prob *= self.robot_trans_model.probability(robot_state, state, action)
            else:
                if next_state[objid] != state[objid]:
                    return 0.0
        return prob

    def sample(self, state, action, **kwargs):
        robot_state = self.robot_trans_model.sample(state, action)
        object_states = {objid : state[objid].copy()
                         for objid in state
                         if objid != robot_state.id}
        object_states[robot_state.id] = robot_state
        return JointState(object_states)

    def get_all_states(self):
        if self.target_states is None:
            raise ValueError("Cannot enumerate over the state space.")

        states = []
        robot_states = self.robot_trans_model.get_all_states()
        for robot_state in robot_states:
            for target_state in self.target_states:
                state = JointState({self.robot_id:robot_state,
                                    target_state.id:target_state},
                                   label="s%d" % (len(states)))
                states.append(state)
        return states


class BasicPolicyModel(pomdp_py.UniformPolicyModel):
    """Default policy model. Pruning
    move actions that bumps into the wall"""
    def __init__(self, actions, robot_trans_model):
        self.move_actions, self.detect_actions, self.declare_actions =\
            self._separate(actions)
        self._legal_moves = {}  # cache
        self.robot_trans_model = robot_trans_model
        super().__init__(actions)

    def _separate(self, actions):
        move_actions = set(a for a in actions if isinstance(a, Move))
        detect_actions = set(a for a in actions if isinstance(a, UseDetector))
        declare_actions = set(a for a in actions if isinstance(a, Declare))
        return move_actions, detect_actions, declare_actions

    def sample(self, state, **kwargs):
        return random.sample(self.get_all_actions(state=state, **kwargs), 1)[0]

    def rollout(self, state, history=None):
        return random.sample(self.get_all_actions(state=state, history=history), 1)[0]

    def valid_moves(self, state):
        robot_id = self.robot_trans_model.robot_id
        if state[robot_id] in self._legal_moves:
            return self._legal_moves[state[robot_id]]
        else:
            robot_pose = state[robot_id]["pose"]
            valid_moves = set(a for a in self.move_actions
                if self.robot_trans_model.sample(state, a)["pose"] != robot_pose)
            self._legal_moves[state[robot_id]] = valid_moves
            return valid_moves

    def get_all_actions(self, state=None, history=None):
        if state is None:
            return self.actions
        else:
            return self.valid_moves(state) | self.detect_actions | self.declare_actions
