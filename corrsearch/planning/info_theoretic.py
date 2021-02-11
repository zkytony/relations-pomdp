import pomdp_py
from corrsearch.models.robot_model import *
from corrsearch.utils import *


class EntropyMinimizationPlanner(pomdp_py.Planner):
    """This is a greedy planner that does not plan
    sequentially and only tries to pick the next best action.

    This is in the spirit of Zeng et al. (2020)
    "Semantic Linking Maps for Active Visual Object Search"
    It says: "We actively search for the target objects
    by generating promising view poses and select the best
    one ranked by a utility function."

    What we will do is: Compute all possible candidate
    viewpoints (i.e. robot poses) given the current robot pose
    (not with respect to the full domain because that actually
     adds actions and is unfair). If the best view point (based on resulting entropy)
    is where the robot is, apply the corresponding best detector.
    Otherwise, perform the move action to move to the best viewpoint.
    Then in the next step, the same computation should result
    in applying the corresponding best detector.

    The expected next belief after applying a detector is an expectation over
    the observations given current belief, estimated by monte-carlo sampling
    `num_samples` number of observations, performing belief update, and then
    averaging the resulting beliefs.
    """

    def __init__(self, robot_trans_model, target_id, num_samples=100, declare_threshold=0.9):
        self.robot_trans = robot_trans_model
        self.target_id = target_id
        self.num_samples = num_samples
        self.declare_threshold = declare_threshold

    @property
    def robot_id(self):
        return self.robot_trans.robot_id

    def sample_next_belief(self, agent, action):
        """
        Returns one sample of next belief given action
        """
        state = agent.belief.random()
        next_state = agent.transition_model.sample(state, action)
        observation = agent.observation_model.sample(next_state, action)
        return agent.belief.update(observation, action)

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

    def compute_expected_entropy(self, belief):
        """
        Compute expected entropy for each detector action,
        over the distribution of observations given current belief
        and the action. Returns a list of tuples (detector_action, entropy)
        """
        result = []
        for a in self.robot_trans.actions:
            if isinstance(a, UseDetector):
                next_belief = self.monte_carlo_next_belief(agent, a, self.num_samples)
                entr = entropy([next_belief[s] for s in next_belief], base=2)
                result.append((a, entr))
        return result


    def plan(self, agent):
        mpe_state = agent.belief.mpe()
        robot_state = mpe_state[self.robot_id]
        robot_pose = robot_state["pose"]

        # First, decide whether to declare. If belief is already
        # greater than some threshold given, then just move towards it.
        if agent.belief[mpe_state] >= self.declare_threshold:
           if robot_pose[:2] == mpe_state[self.target_id]["loc"]:
               return Declare()
           else:
               min_dist = float("inf")
               next_move_action = None
               for a in self.robot_trans.actions:
                   if isinstance(a, Move):
                       next_belief = self.sample_next_belief(agent, a)
                       next_robot_pose = next_belief.mpe()[self.robot_id]["pose"]
                       target_loc = mpe_state[self.target_id]["loc"]
                       distance = euclidean_dist(next_robot_pose[:2], target_loc)
                       if distance < min_dist:
                           min_dist = distance
                           next_move_action = a
               return next_move_action

        # mapping from (pose, move_action, detector_action) to resulting belief entropy
        entropy_map = {}

        # First, compute the entropy of expected beliefs if the
        # robot stays where it is
        expected_entropies = self.compute_expected_entropy(agent.belief)
        for detector_action, entr in expected_entropies:
            entropy_map[(robot_pose, None, detector_action)] = entr

        # Next, compute the entropy of expected beliefs if the robot
        # moves one step, and then applies detector
        for action in self.robot_trans.actions:
            if isinstance(action, Move):
                # incorporating the movement
                next_belief = self.sample_next_belief(agent, action)
                expected_entropies = self.compute_expected_entropy(next_belief)
                next_robot_pose = next_belief.mpe()[self.robot_id]["pose"]
                for detector_action, entr in expected_entropies:
                    entropy_map[(next_robot_pose, action, detector_action)] = entr

        # Now, pick the pair of pose and action with lowest entropy
        pose, move_action, detector_action = min(entropy_map, entropy_map.get)
        if move_action is not None:
            return move_action
        else:
            return detector_action
