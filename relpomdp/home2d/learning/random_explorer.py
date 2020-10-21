import pomdp_py
import random

class RandomPlanner(pomdp_py.Planner):
    def __init__(self, robot_id, legal_motions):
        self.robot_id = robot_id
        self.legal_motions = legal_motions

    def plan(self, agent):
        agent_state = agent.belief.mpe().object_states[self.robot_id]
        valid_motions = self.legal_motions[agent_state["pose"][:2]]
        return random.sample(valid_motions, 1)[0]
