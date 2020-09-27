import pomdp_py
import copy
from relpomdp.object_search.agent import *
import relpomdp.oopomdp.framework as oopomdp
from relpomdp.planning.subgoal import Subgoal

# One more condition-effect pair for transition
class AchievingSubgoals(oopomdp.Condition):
    """
    This is used to transition the `subgoals_achieved` attribute
    in RobotStateWithSubgoals. Returns true whenever a new subgoal
    is achieved.
    """
    def __init__(self, ids, subgoals):
        # The subgoal here is the room type
        self.ids = ids
        self.subgoals = subgoals

    def satisfy(self, state, action, *args):
        """It is assumed that if action is move, `state` has
        already incorporated that action"""
        robot_id = self.ids["Robot"]
        robot_state = state.object_states[robot_id]
        subgoals_status = []  # list of tuples (subgoal_name, status)
        for goal_name, status in robot_state["subgoals"]:
            tup = (goal_name, status)
            if status == Subgoal.IP:
                # If still in progress, then it may change
                if self.subgoals[goal_name].achieve(state, action):
                    tup = (goal_name, Subgoal.SUCCESS)
                elif self.subgoals[goal_name].fail(state, action):
                    tup = (goal_name, Subgoal.FAIL)
                subgoals_status.append(tup)
        if len(subgoals_status) == 0:
            return False, []
        else:
            return True, subgoals_status

class UpdateSubgoalStatus(oopomdp.DeterministicTEffect):
    """
    This is used to transition the `subgoals_achieved` attribute
    in RobotStateWithSubgoals.
    """
    def __init__(self, ids):
        self.ids = ids
        super().__init__("constant")  # ? not really a reason to name the type this way
        
    def mpe(self, state, action, subgoal_status):
        """Returns an OOState after applying this effect on `state`"""
        robot_id = self.ids["Robot"]
        next_state = state  # no need to call .copy because state is already a copy
        next_robot_state = next_state.object_states[robot_id]
        next_robot_state["subgoals"] = subgoal_status
        return next_state
