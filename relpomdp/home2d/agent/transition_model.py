import relpomdp.oopomdp.framework as oopomdp
from relpomdp.home2d.domain.relation import *
from relpomdp.home2d.domain.action import *

class DeclareFound(Action):
    """Declare Found action: Declares an object to be found,
    when the robot is on top of it."""
    def __init__(self):
        super().__init__("declare-found")

class CanDeclareFound(oopomdp.Condition):
    """Pick up condition"""
    def __init__(self, robot_id, target_id):
        self.robot_id = robot_id
        self.target_id = target_id

    def satisfy(self, state, action):
        if not isinstance(action, DeclareFound):
            return False
        robot_state = state.object_states[self.robot_id]
        target_state = state.object_states[self.target_id]
        if is_on(robot_state, target_state)\
           and not target_state.get("is_found", False):
            return True, self.target_id
        return False

class DeclareFoundEffect(oopomdp.DeterministicTEffect):
    """Pick up effect: Deterministically pick up"""
    def __init__(self):
        super().__init__("declare-found")

    def mpe(self, state, action, picking_objid):
        """Returns an OOState after applying this effect on `state`"""
        next_state = state
        next_state.object_states[picking_objid]["is_found"] = True
        return next_state
