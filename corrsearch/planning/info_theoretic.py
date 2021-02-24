import pomdp_py
from corrsearch.models.robot_model import *
from corrsearch.utils import *


class EntropyMinimizationPlanner(pomdp_py.Planner):
    """This is a greedy planner that does not plan sequentially and only tries to
    pick the next best action.

    This is in the spirit of Zeng et al. (2020) "Semantic Linking Maps for
    Active Visual Object Search" It says: "We actively search for the target
    objects by generating promising view poses and select the best one ranked by
    a utility function."

    What we will do is: Among all detectors, pick the one that leads to the
    lowest expected entropy. If the entropy improvement falls below a threshold,
    then try to move closer to the most likely location.

    The expected next belief after applying a detector is an expectation over
    the observations given current belief, estimated by monte-carlo sampling
    `num_samples` number of observations, performing belief update, and then
    averaging the resulting beliefs.
    """

    def __init__(self, num_samples=100, declare_threshold=0.9, entropy_improvement_threshold=1e-3):
        self.num_samples = num_samples
        self.declare_threshold = declare_threshold
        self.entropy_improvement_threshold = entropy_improvement_threshold

    def sample_next_belief(self, agent, action):
        """
        Returns one sample of next belief given action
        """
        state = agent.belief.random()
        next_state = agent.transition_model.sample(state, action)
        observation = agent.observation_model.sample(next_state, action)
        return agent.belief.update(agent, observation, action)

    def monte_carlo_next_belief(self, agent, action, num_samples):
        belief_hist = {s:0.0 for s in agent.belief}
        for i in range(num_samples):
            next_belief = self.sample_next_belief(agent, action)
            for state in belief_hist:
                belief_hist[state] += next_belief[state]
        # normalize
        total_prob = sum(belief_hist.values())
        for state in belief_hist:
            belief_hist[state] /= total_prob
        return pomdp_py.Histogram(belief_hist)

    def compute_expected_entropy(self, agent, belief):
        """
        Compute expected entropy for each detector action,
        over the distribution of observations given current belief
        and the action. Returns a list of tuples (detector_action, entropy)
        """
        result = []
        actions = agent.policy_model.actions
        for a in actions:
            if isinstance(a, UseDetector):
                next_belief = self.monte_carlo_next_belief(agent, a, self.num_samples)
                entr = entropy([next_belief[s] for s in next_belief], base=2)
                result.append((a, entr))
        return result

    def move_towards_highest_belief(self, agent, mpe_state):
        """Returns a move action in the direction of the highest belief state."""
        robot_trans = agent.transition_model.robot_trans_model
        target_id = agent.belief.target_id
        robot_id = robot_trans.robot_id
        robot_state = mpe_state[robot_id]
        robot_pose = robot_state["pose"]

        min_dist = float("inf")
        next_move_action = None
        for a in robot_trans.actions:
            if isinstance(a, Move):
                next_belief = self.sample_next_belief(agent, a)
                next_robot_pose = next_belief.mpe()[robot_id]["pose"]
                target_loc = mpe_state[target_id]["loc"]
                distance = euclidean_dist(next_robot_pose[:2], target_loc)
                if distance < min_dist:
                    min_dist = distance
                    next_move_action = a
        return next_move_action


    def plan(self, agent):
        mpe_state = agent.belief.mpe()
        target_id = agent.belief.target_id
        robot_id = agent.transition_model.robot_trans_model.robot_id
        robot_state = mpe_state[robot_id]
        robot_pose = robot_state["pose"]

        # First, decide whether to declare. If belief is already
        # greater than some threshold given, then just move towards it.
        if agent.belief[mpe_state] >= self.declare_threshold:
           if robot_pose[:2] == mpe_state[target_id]["loc"]:
               return Declare()
           else:
               return self.move_towards_highest_belief(agent, mpe_state)

        # Compute the entropy of expected beliefs if the robot stays where it is
        expected_entropies = self.compute_expected_entropy(agent, agent.belief)
        detector_action, min_entr = min(expected_entropies, key=lambda t: t[1])

        current_entr = entropy([agent.belief[s] for s in agent.belief], base=2)
        if current_entr - min_entr >= self.entropy_improvement_threshold:
            # Good enough improvement. Apply detect action
            return detector_action
        else:
            # Improvement not good enough. Move
            return self.move_towards_highest_belief(agent, mpe_state)
