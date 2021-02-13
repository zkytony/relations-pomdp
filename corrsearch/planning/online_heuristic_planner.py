"""
Heuristic planner that plans as follows:

- Selects a subset of K detectors to plan with based on pessimistic value estimate
- Uses a heuristic rollout policy to select detectors based on expected number of
  object detections.
"""

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

    def value_lower_bound(self, target_id, belief, reward_model):
        """Returns a lower bound on the value at the belief"""
        target_belief = belief.obj(target_id)
        btarget_max = max(target_belief[s] for s in target_belief)
        return (reward_model.rmax - reward_model.rmin) * btarget_max\
            + reward_model.rmin

    def choose_detectors(self, agent):
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
        detectors_sorted = [(d,val) for val, d in sorted(zip(vals, detectors))]
        return detectors_sorted[:self.k]

    def plan(self, agent):
        # First, choose a subset of detectors
        detectors_vals = self.choose_detectors(agent)
        detector_valmap = {d:v for d, v in detectors_vals}

        # Create an agent, for planning, with heuristic policy model
        robot_id = agent.belief.robot_id
        target_id = agent.belief.target_id
        default_policy_model = agent.policy_model
        heuristic_policy_model = HeuristicRollout(
            robot_id, target_id,
            default_policy_model,
            [UseDetector(d) for d in detector_valmap],
            agent.transition_model.robot_trans_model,
            agent.observation_model,
            num_visits_init=self.params.get("ap_num_visits_init", 10),
            val_init=self.params.get("ap_val_init", 100))
        tmp_agent = pomdp_py.Agent(agent.belief,
                                   heuristic_policy_model,
                                   agent.transition_model,
                                   agent.observation_model,
                                   agent.reward_model)

        # Insert Q-Node for root node with initial value equal to the lower
        # bound, returned by `choose_detectors
        tmp_agent.tree = pomdp_py.RootVNode(self.params["num_visits_init"],
                                            float("-inf"),
                                            agent.history)
        for action in tmp_agent.valid_actions():
            if isinstance(action, UseDetector):
                val_init = detector_valmap[action.detector_id]
            else:
                val_init = self.value_lower_bound(target_id, tmp_agent.belief,
                                                  tmp_agent.reward_model)

            if tmp_agent.tree[action] is None:
                tmp_agent.tree[action] = pomdp_py.QNode(self.params.get("num_visits_init", 0),
                                                        val_init)

        # Create planner, plan action
        planner = pomdp_py.POUCT(max_depth=self.params["max_depth"],
                                 num_sims=self.params["max_depth"],
                                 discount_factor=self.params["discount_factor"],
                                 num_visits_init=self.params.get("num_visits_init", 0),
                                 exploration_const=self.params["exploration_const"],
                                 rollout_policy=tmp_agent.policy_model,
                                 action_prior=tmp_agent.policy_model.action_prior)
        action = planner.plan(tmp_agent)
        agent.tree = tmp_agent.tree
        return action


class HeuristicRollout(BasicPolicyModel):
    """Rollout policy for heuristic planner.
    This is initialized for the agent with CURRENT belief,
    to plan for the NEXT step"""
    def __init__(self,
                 robot_id,
                 target_id,
                 default_policy_model,
                 detector_actions,
                 robot_trans_model,
                 observation_model,
                 num_visits_init=10,
                 val_init=100):
        self.robot_id = robot_id
        self.target_id = target_id

        self.robot_trans_model = robot_trans_model

        assert not isinstance(default_policy_model, HeuristicRollout)
        self._default_policy_model = default_policy_model
        all_actions = default_policy_model.actions
        move_actions, _, declare_actions = self._separate(all_actions)
        actions_by_type = {"move": set(move_actions),
                   "detect": set(detector_actions),
                   "declare": set(declare_actions)}

        self.action_prior = HeuristicActionPrior(robot_id, target_id,
                                                 robot_trans_model,
                                                 actions_by_type,
                                                 observation_model,
                                                 num_visits_init=num_visits_init,
                                                 val_init=val_init)
        super().__init__(actions_by_type["move"] | actions_by_type["detect"] | actions_by_type["declare"])

    def sample(self, state, **kwargs):
        """For sampling, use the given agent's policy model directly."""
        return default_policy_model.sample(state, **kwargs)

    def rollout(self, state, history):
        """For rollout, use a policy from an action prior"""
        # Obtain preference and returns the action in it.
        if not hasattr(self, "action_prior"):
            raise ValueError("PreferredPolicyModel is not assigned an action prior.")
        preferences = self.action_prior.get_preferred_actions(state, history)
        if len(preferences) > 0:
            return random.sample(preferences, 1)[0][0]
        else:
            return self.sample(state)


class HeuristicActionPrior(pomdp_py.ActionPrior):
    def __init__(self, robot_id, target_id,
                 robot_trans_model, actions_by_type, observation_model,
                 num_visits_init=10, val_init=100):
        self.robot_id = robot_id
        self.target_id = target_id
        self.num_visits_init = num_visits_init
        self.val_init = val_init
        self.robot_trans_model = robot_trans_model
        self.actions_by_type = actions_by_type
        self.observation_model = observation_model

    def get_preferred_actions(self, state, history):
        preferences = set()

        # Preference of move
        move_actions = self.actions_by_type["move"]
        preferred_move = min(move_actions,
            key=lambda move_action: euclidean_dist(self.robot_trans_model.sample(state, move_action)["loc"],
                                                   state[self.robot_id]["loc"]))
        preferences.add((preferred_move, self.num_visits_init, self.val_init))

        for detect_action in self.actions_by_type["detect"]:
            z = self.observation_model.sample(state, detect_action)
            for objid in z.object_obzs:
                if not isinstance(z[objid], NullObz):
                    preferences.add((detect_action, self.num_visits_init, self.val_init))
                    break

        return preferences
