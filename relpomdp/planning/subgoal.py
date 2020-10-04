
class Subgoal:
    """Subgoal is like a condition that is True when it is achieved.
    Its achieve depends on s,a and it triggers a state transition."""
    # Status
    IP = "IP"
    SUCCESS = "SUCCESS"
    FAIL = "FAIL"
    
    def __init__(self, name):
        """
        status can be "IP": In Progress; "SUCCESS": achieved; or "FAIL": Failed
        """
        self.name = name
    def achieve(self, next_state, action):
        pass
    def fail(self, next_state, action):
        pass
    def __str__(self):
        return "%s(%s)" % (self.__class__.__name__, self.name)
    def trigger_success(self, robot_state, action, observation):
        """Called when this subgoal is achieved. Returns
        the next subgoal (if needed); This is assumed to
        be called during planner update."""
        return None
    def trigger_fail(self, robot_state, action, observation):
        """Called when this subgoal is failed. Returns
        the next subgoal (if needed)"""        
        return None
