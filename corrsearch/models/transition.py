import pomdp_py
from corrsearch.objects import *

class SearchTransitionModel(pomdp_py.TransitionModel):
    def __init__(self, robot_id, robot_trans_model):
        self.robot_id = robot_id
        self.robot_trans_model = robot_trans_model

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
