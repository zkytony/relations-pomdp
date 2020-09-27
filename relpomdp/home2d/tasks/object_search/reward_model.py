import pomdp_py
from relpomdp.home2d.action import *

class RewardModel(pomdp_py.RewardModel):
    def __init__(self, ids):
        self.ids = ids
    
    def sample(self, state, action, next_state, **kwargs):
        return self.argmax(state, action, next_state)
    
    def argmax(self, state, action, next_state, **kwargs):
        """
        argmax(self, state, action, next_state, **kwargs)
        Returns the most likely reward"""
        # Reward is 1 if picked up a target, -1 if wrong. -0.1 otherwise
        if isinstance(action, Pickup):
            all_found = True
            for objid in self.ids["Target"]:
                found = state.object_states[objid]["is_found"]
                next_found = next_state.object_states[objid]["is_found"]
                if not found:
                    all_found = False
                if not found and next_found:
                    return 100.0
            # Didn't pick up anything new
            if not all_found:
                return -100.0
            else:
                return -1
        return -1
