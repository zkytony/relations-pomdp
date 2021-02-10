import pomdp_py
from corrsearch.objects import *

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
