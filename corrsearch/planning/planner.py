import pomdp_py
from corrsearch.models import *

def belief_reward(belief, reward_model, action):
    return sum(belief[state] * reward_model.sample(state, action)
               for state in belief)

class RSSPolicyModel(pomdp_py.UniformPolicyModel):
    def __init__(self, actions):
        super().__init__(actions)

    @property
    def move_actions(self):
        raise NotImplementedError

    @property
    def detect_actions(self):
        raise NotImplementedError

    @property
    def declare_actions(self):
        raise NotImplementedError


class RSSPlanner(pomdp_py.Planner):

    """
    The RSSPlanner is our algorithm. It basically runs POMCP
    with a heuristic of:
    - action subset for planning
    - rollout
    """

    def __init__(self, search_problem, gamma=0.95, **solver_params):
        self.SP = search_problem
        self.gamma = gamma
        self.pomcp = pomdp_py.POMCP(**solver_params)

    def value_lower_bound(self, belief, agent):
        """Returns a lower bound on the value at the belief"""
        target_id = self.SP.target_object
        target_belief = belief.obj(target_id)
        btarget_max = max(target_belief[s] for s in target_belief)
        return (agent.reward_model.rmax - agent.reward_model.rmin) * btarget_max\
            + agent.reward_model.rmin

    def choose_detectors(self, agent, k=2, num_zsamples=10):
        """
        The Agent should maintain belief only about the target object.
        The question is to choose a set of detectors to plan with.
        """
        detectors = self.SP.robot_model.detectors
        target_id = self.SP.target_object
        vals = []
        for d in detectors:
            action = UseDetector(d)
            immediate_reward = belief_reward(agent.belief.obj(target_id),
                                             agent.reward_model,
                                             action)

            # estimate future reward
            expected_future_reward = 0.0
            for i in range(num_zsamples):
                state = agent.belief.sample()
                next_state = agent.transition_model.sample(state, action)
                z = agent.observation_model.sample(next_state, action)
                next_belief = agent.belief.update(z, action)
                expected_future_reward +=\
                    agent.belief[state]\
                    * agent.observation_model.probability(z, next_state, action)\
                    * agent.transition_mode.probability(next_state, state, action)\
                    * self.value_lower_bound(next_belief, agent)
            vals.append(immediate_reward + self.gamma * expected_future_reward)
        detectors_sorted = [d for _, d in sorted(zip(vals, detectors))]
        return detectors_sorted[:k]

    def plan(self, agent, k=2, num_zsamples=10):
        detectors = self.choose_detectors(agent, k=k, num_zsamples=num_zsamples)

        actions = agent.policy_model.move_actions\
                  | set(UseDetector(d) for d in detectors)\
                  | agent.policy_model.declare_actions

        policy_model = RSSPolicyModel(actions)
        setattr(agent, "_policy_model", policy_model)
        return self.pomcp.plan(agent)
