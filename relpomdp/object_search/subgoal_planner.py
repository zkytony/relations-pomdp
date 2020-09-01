import pomdp_py
import copy
from relpomdp.object_search.state import *
from relpomdp.object_search.agent import *
from relpomdp.object_search.reward_model import RewardModel
import relpomdp.oopomdp.framework as oopomdp

class Subgoal:
    def __init__(self, name):
        self.name = name
    def achieved(self, state, action, next_state):
        pass

class ReachRoomSubgoal(Subgoal):
    def __init__(self, ids, room_type, grid_map):
        self.ids = ids
        self.grid_map = grid_map
        self.room_type = room_type
        super().__init__("Reach-%s" % room_type)
        
    def achieved(self, state, action):
        robot_id = self.ids["Robot"]        
        room_name = self.grid_map.room_of(state.object_states[robot_id].pose[:2])
        room = self.grid_map.rooms[room_name]
        return room.room_type == self.room_type


class RobotStateWithSubgoals(RobotState):
    def __init__(self, pose, camera_direction, subgoals_achieved=tuple()):
        PoseState.__init__(self,
                           "Robot",
                           {"pose":pose,  # x,y,th
                            "camera_direction": camera_direction,
                            "subgoals_achieved": subgoals_achieved})
    def copy(self):
        return self.__class__(tuple(self["pose"]), self.camera_direction,
                              tuple(self["subgoals_achieved"]))
    
    def to_state_without_subgoals(self):
        return RobotState(tuple(self["pose"]),
                          self.camera_direction)
    @classmethod
    def from_state_without_subgoals(cls, robot_state, subgoals_achieved=tuple()):
        return RobotStateWithSubgoals(tuple(robot_state.pose),
                                      robot_state.camera_direction,
                                      subgoals_achieved=subgoals_achieved)


# One more condition-effect pair for transition
class AchievingSubgoal(oopomdp.Condition):
    def __init__(self, ids, subgoals, grid_map):
        # The subgoal here is the room type
        self.ids = ids
        self.subgoals = subgoals  # a dictionary {name -> Subgoal}
        self.grid_map = grid_map

    def satisfy(self, state, action, *args):
        robot_id = self.ids["Robot"]
        robot_state = state.object_states[robot_id]
        subgoals_achieved = []
        for goal_name in self.subgoals:
            # The subgoal here is the room type
            if goal_name not in robot_state["subgoals_achieved"]:
                if self.subgoals[goal_name].achieved(state, action):
                    subgoals_achieved.append(goal_name)
        if len(subgoals_achieved) == 0:
            return False, []
        else:
            return True, subgoals_achieved

class AddSubgoal(oopomdp.DeterministicTEffect):
    def __init__(self, ids):
        self.ids = ids
        super().__init__("constant")  # ? not really a reason to name the type this way
        
    def mpe(self, state, action, new_subgoals_achieved):
        """Returns an OOState after applying this effect on `state`"""
        robot_id = self.ids["Robot"]
        next_state = state.copy()
        current_subgoal_achieved = state.object_states[robot_id]["subgoals_achieved"]
        next_state.object_states[robot_id]["subgoals_achieved"] =\
            tuple(set(current_subgoal_achieved) | set(new_subgoals_achieved))
        return next_state    

class SubgoalRewardModel(RewardModel):
    def argmax(self, state, action, next_state, **kwargs):
        """
        argmax(self, state, action, next_state, **kwargs)
        Returns the most likely reward"""
        # Reward is 1 if picked up a target, -1 if wrong. -0.1 otherwise
        robot_id = self.ids["Robot"]
        robot_state = state.object_states[robot_id]
        next_robot_state = next_state.object_states[robot_id]
        reward = 0
        if len(robot_state["subgoals_achieved"]) < len(next_robot_state["subgoals_achieved"]):
            # achieved more subgoals, good
            reward = 50.0
        reward += super().argmax(state, action, next_state, **kwargs)
        return reward

        
class SubgoalPlanner(pomdp_py.Planner):
    """The subgoal planner works by replacing the robot state in
    the agent's belief by a state that has an additional attribute,
    "subgoals_achieved" which tracks which subgoals are achieved"""
    def __init__(self, ids, subgoals, grid_map, planner):
        self.ids = ids
        self.subgoals = subgoals
        self.grid_map = grid_map
        self._robot_state_with_subgoals = None
        self._planner = planner

    def plan(self, agent):
        robot_id = self.ids["Robot"]
        target_id = self.ids["Target"][0]
        robot_state = agent.belief.mpe().object_states[robot_id]        
        if self._robot_state_with_subgoals is None:
            self._robot_state_with_subgoals =\
                RobotStateWithSubgoals.from_state_without_subgoals(robot_state)

        # Create a temporary agent, with subgoal-aware transition/reward models
        belief = pomdp_py.OOBelief({
            robot_id:pomdp_py.Histogram({self._robot_state_with_subgoals.copy():1.0}),
            target_id:agent.belief.object_beliefs[target_id]})
        transition_model = oopomdp.OOTransitionModel(
            set(agent.transition_model.cond_effects)\
            | {(AchievingSubgoal(self.ids, self.subgoals, self.grid_map),
                AddSubgoal(self.ids))})
        reward_model = SubgoalRewardModel(self.ids)
        tmp_agent = pomdp_py.Agent(belief,
                                   agent.policy_model,
                                   transition_model,
                                   agent.observation_model,
                                   reward_model)
        # Plan accordingly
        action = self._planner.plan(tmp_agent)
        
        # Record the subgoals achieved if execute this action; Note that the action
        # is not executed right now. We are just recording the subgoals
        next_mpe_state = transition_model.sample(tmp_agent.belief.mpe(), action)
        self._robot_state_with_subgoals = next_mpe_state.object_states[robot_id].copy()
        
        return action

    def update(self, agent, action, observation):
        # Now agent's belief has been updated
        robot_id = self.ids["Robot"]
        robot_state = agent.belief.mpe().object_states[robot_id]
        if self._robot_state_with_subgoals is not None:
            assert robot_state == self._robot_state_with_subgoals.to_state_without_subgoals(),\
                "After executing action, robot_state != robot_state_with_subgoals"
        self._planner.update(agent, action, observation)
        
        
            
