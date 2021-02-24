import pomdp_py
from corrsearch.objects import *
from corrsearch.models import *
import random

class RandomPlanner(pomdp_py.Planner):
    """
    Completely Random Planner.
    Takes Declare action if belief at current location
    is greater than a given threshold
    """
    def __init__(self, declare_threshold=0.9):
        self.declare_threshold = declare_threshold

    def plan(self, agent):
        target_id = agent.belief.target_id
        robot_id = agent.transition_model.robot_trans_model.robot_id
        random_state = agent.belief.random()

        # Query belief at robot location
        state = JointState({robot_id: random_state[robot_id],
                            target_id: LocObjState(target_id, random_state[target_id].objclass,
                                                   {"loc": random_state[robot_id].loc})})
        if agent.belief[state] >= self.declare_threshold:
            return Declare()
        else:
            actions = agent.policy_model.actions
            return random.sample(set(actions) - {Declare()}, 1)[0]
