"""
Heuristic planner that plans as follows:

- Selects a subset of K detectors to plan with based on pessimistic value estimate
- Uses a heuristic rollout policy to select detectors based on expected number of
  object detections.
"""

import random
import pomdp_py
from corrsearch.models import *
from corrsearch.utils import *

def monte_carlo_belief_reward(belief, reward_model, transition_model, action, num_samples=30):
    reward = 0
    for i in range(num_samples):
        state = belief.random()
        next_state = transition_model.sample(state, action)
        reward += belief[state] * transition_model.probability(next_state, state, action)\
                  * reward_model.sample(state, action, next_state)
    return reward

class HeuristicSequentialPlanner(pomdp_py.Planner):

    def __init__(self,
                 k=2,
                 num_rsamples=30,
                 num_zsamples=100,
                 gamma=0.95,
                 init_qvalue_lower_bound=False,
                 **params):
        """
        Args:

        num_rsamples (int): Number of samples to estimate belief reward
        num_zsamples (int): Number of samples to estimate next belief value lower bound
        """
        self.k = k
        self.num_zsamples = num_zsamples
        self.num_rsamples = num_rsamples
        self.gamma = gamma
        self.params = params
        self.init_qvalue_lower_bound = init_qvalue_lower_bound
        self._last_action_observation = None

    def value_lower_bound(self, target_id, belief, reward_model):
        """Returns a lower bound on the value at the belief"""
        target_belief = belief.obj(target_id)
        btarget_max = max(target_belief[s] for s in target_belief)
        return (reward_model.rmax - reward_model.rmin) * btarget_max\
            + reward_model.rmin

    def detector_values(self, agent):
        """
        The Agent should maintain belief only about the target object.
        The question is to choose a set of detectors to plan with.
        """
        detectors = agent.observation_model.detectors
        target_id = agent.belief.target_id
        vals = []
        for d in detectors:
            action = UseDetector(d)
            immediate_reward = monte_carlo_belief_reward(agent.belief,
                                                         agent.reward_model,
                                                         agent.transition_model,
                                                         action,
                                                         num_samples=self.num_rsamples)

            # estimate future reward
            expected_future_reward = 0.0
            for i in range(self.num_zsamples):
                state = agent.belief.random()
                next_state = agent.transition_model.sample(state, action)
                z = agent.observation_model.sample(next_state, action)
                next_belief = agent.belief.update(agent, z, action)
                expected_future_reward +=\
                    agent.belief[state]\
                    * agent.observation_model.probability(z, next_state, action)\
                    * agent.transition_model.probability(next_state, state, action)\
                    * self.value_lower_bound(target_id, next_belief, agent.reward_model)
            vals.append(immediate_reward + self.gamma * expected_future_reward)
        detector_valmap = {d:v for v, d in sorted(zip(vals, detectors))}
        return detector_valmap


    def plan(self, agent):
        # First, choose a subset of detectors
        if self.k < len(agent.observation_model.detectors):
            detector_valmap = self.detector_values(agent)
            chosen_detectors = list(sorted(detector_valmap, key=detector_valmap.get, reverse=True))[:self.k]
            detector_valmap = {d:detector_valmap[d] for d in chosen_detectors}
            print("Detectors {} chosen from {}".format(list(detector_valmap.keys()),
                                                       agent.observation_model.detectors))
        else:
             chosen_detectors = agent.observation_model.detectors

        # Create an agent, for planning, with heuristic policy model
        robot_id = agent.belief.robot_id
        target_id = agent.belief.target_id
        default_policy_model = agent.policy_model
        heuristic_policy_model = HeuristicRollout(
            robot_id, target_id,
            agent.policy_model,
            agent.transition_model,
            agent.observation_model,
            agent.reward_model)
        tmp_agent = pomdp_py.Agent(agent.belief,
                                   heuristic_policy_model,
                                   agent.transition_model,
                                   agent.observation_model,
                                   agent.reward_model)

        # Build new tree
        tmp_agent.tree = pomdp_py.RootVNode(self.params["num_visits_init"],
                                            float("-inf"),
                                            agent.history)
        cur_belief_val_lower_bound = self.value_lower_bound(target_id, tmp_agent.belief,
                                                            tmp_agent.reward_model)

        for action in tmp_agent.valid_actions(state=tmp_agent.belief.mpe(),
                                              history=self._last_action_observation):
            if self.init_qvalue_lower_bound:
                if isinstance(action, UseDetector):
                    val_init = detector_valmap[action.detector_id]
                else:
                    val_init = cur_belief_val_lower_bound
            else:
                val_init = self.params.get("val_init", 0)

            if tmp_agent.tree[action] is None:
                tmp_agent.tree[action] = pomdp_py.QNode(self.params.get("num_visits_init", 0),
                                                        val_init)

        # Create planner, plan action
        planner = pomdp_py.POUCT(max_depth=self.params["max_depth"],
                                 num_sims=self.params["num_sims"],
                                 discount_factor=self.params["discount_factor"],
                                 num_visits_init=self.params.get("num_visits_init", 0),
                                 exploration_const=self.params["exploration_const"],
                                 rollout_policy=tmp_agent.policy_model)
        action = planner.plan(tmp_agent)
        agent.tree = tmp_agent.tree
        return action

    def update(self, agent, real_action, real_observation):
        self._last_action_observation = [(real_action, real_observation)]


class HeuristicRollout(pomdp_py.RolloutPolicy):
    """Rollout policy for heuristic planner.
    This is initialized for the agent with CURRENT belief,
    to plan for the NEXT step"""
    def __init__(self,
                 robot_id,
                 target_id,
                 original_policy_model,
                 transition_model,
                 observation_model,
                 reward_model):
        self.robot_id = robot_id
        self.target_id = target_id
        self.original_policy_model = original_policy_model
        self.observation_model = observation_model
        self.reward_model = reward_model
        self.transition_model = transition_model
        self.robot_trans_model = transition_model.robot_trans_model
        self._cache = {}

    def sample(self, state, **kwargs):
        return self.original_policy_model.sample(state, **kwargs)

    def get_all_actions(self, *args, **kwargs):
        return self.original_policy_model.get_all_actions(*args, **kwargs)

    def rollout(self, state, history):
        """For rollout, use a policy from an action prior"""
        # Obtain preference and returns the action in it.
        key_ = (state, history[-1])
        if key_ in self._cache:
            candidates = self._cache[key_]
        else:
            candidates = []
            move_actions = self.original_policy_model.valid_moves(state)
            preferred_move = min(move_actions,
                key=lambda move_action: euclidean_dist(self.robot_trans_model.sample(state, move_action)["loc"],
                                                       state[self.target_id]["loc"]))
            candidates.append(preferred_move)
            for detect_action in self.original_policy_model.detect_actions:
                detector = self.observation_model.detectors[detect_action.detector_id]
                z = self.observation_model.sample(state, detect_action)
                for objid in z.object_obzs:
                    if not isinstance(z[objid], NullObz):
                        candidates.append(detect_action)
                        break
            if isinstance(history[-1][0], UseDetector):
                # Declare follows detect; This is what is done in the basic rollout policy too
                if Declare(state[self.target_id]) in self.original_policy_model.declare_actions:
                    candidates.append(Declare(state[self.target_id]))
                else:
                    next_state = self.transition_model.sample(state, Declare())
                    if self.reward_model.sample(state, Declare(), next_state)\
                       == self.reward_model.rmax:
                        candidates.append(Declare())
            self._cache[key_] = candidates
        return random.sample(candidates, 1)[0]
