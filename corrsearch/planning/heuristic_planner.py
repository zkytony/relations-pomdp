"""
Plans with POUCT + heuristic rollout
"""
import pomdp_py

class PreferredRollout(pomdp_py.RolloutPolicy):
    """Heuristic rollout policy. No action prior.
    Just heuristic rollout"""

    def __init__(self, action_prior, default_rollout):
        """
        Args:
            default_policy_model (RolloutPolicy): Policy model to use,
                if heuristic is not applied
        """
        self.action_prior = action_prior
        self.default_rollout = default_rollout

    def rollout(self, state, history):
        # Obtain preference and returns the action in it.
        if not hasattr(self, "action_prior"):
            raise ValueError("PreferredPolicyModel is not assigned an action prior.")
        preferences = self.action_prior.get_preferred_actions(state, history)
        if len(preferences) > 0:
            return random.sample(preferences, 1)[0][0]
        else:
            return self.default_rollout.rollout(state, history)

    def add_action_prior(self, action_prior_class, *args, **kwargs):
        self.action_prior = action_prior_class.build(self, *args, **kwargs)


class CorrActionPrior(pomdp_py.ActionPrior):
    def __init__(self, robot_id, target_id,
                 detectors, num_visits_init=1, val_init=100):
        """Args:
        robot_id (int) Id of robot
        target_id (int) Id of target
        detectors (list or dict) list of DetectorModel objects
            or dict mapping form id to DetectorModel"""
        self.robot_id = robot_id
        self.target_id = target_id
        if type(detector_models) == list:
            self.detectors = {d.id for d in detector_models}
        else:
            self.detectors = detectors

    def get_preferred_actions(self, state, history):
        pass



class HeuristicPlanner(pomdp_py.Planner):

    def __init__(self, **planner_config):
        self._pouct = pomdp_py.POUCT(**planner_config)

    def plan(self, agent):
        pass
