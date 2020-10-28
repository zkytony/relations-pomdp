import pomdp_py
from relpomdp.home2d.agent.transition_model import Pickup

class PickupRewardModel(pomdp_py.RewardModel):
    """
    Reward model for search item task
    """
    def __init__(self, robot_id, target_id):
        self.robot_id = robot_id
        self.target_id = target_id

    def sample(self, state, action, next_state, **kwargs):
        return self.argmax(state, action, next_state)

    def argmax(self, state, action, next_state, **kwargs):
        """
        argmax(self, state, action, next_state, **kwargs)
        Returns the most likely reward"""
        # Reward is 100 if picked up a target, -100 if wrong. -1 otherwise
        if isinstance(action, Pickup):
            found = state.object_states[self.target_id].get("is_found", False)
            next_found = next_state.object_states[self.target_id].get("is_found", False)
            if next_found:
                if not found:
                    return 100.0
                else:
                    return -1.0
            else:
                return -100.0
        return -1.0


class ReachRewardModel(pomdp_py.RewardModel):
    def __init__(self, robot_id, target_id):
        self.robot_id = robot_id
        self.target_id = target_id

    def sample(self, state, action, next_state, **kwargs):
        return self.argmax(state, action, next_state)

    def argmax(self, state, action, next_state, **kwargs):
        """
        argmax(self, state, action, next_state, **kwargs)
        Returns the most likely reward"""
        robot_state = state.object_states[self.robot_id]
        next_robot_state = state.object_states[self.robot_id]
        target_state = state.object_states[self.target_id]
        assert target_state["pose"] == next_state.object_states[self.target_id]["pose"]
        if next_robot_state["pose"][:2] == target_state["pose"]:
           if robot_state["pose"][:2] != target_state["pose"]:
               return 100.0
           else:
               return -1.0
        else:
            return -1.0
