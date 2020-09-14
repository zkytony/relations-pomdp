import pomdp_py
import relpomdp.oopomdp.framework as oopomdp
from relpomdp.object_search.state import *
from relpomdp.planning.subgoal import Subgoal
from relpomdp.object_search.reward_model import RewardModel
from relpomdp.object_search.subgoal_cond_effect import *

class RobotStateWithSubgoals(RobotState):
    """This is needed for the robot to keep track
    of subgoals achieved so that it does not achieve a subgoal twice"""
    def __init__(self, pose, camera_direction, subgoals=tuple()):
        PoseState.__init__(self,
                           "Robot",
                           {"pose":pose,  # x,y,th
                            "camera_direction": camera_direction,
                            "subgoals": subgoals})   # stores (subgoal_name, status) tuples
    def copy(self):
        return self.__class__(tuple(self["pose"]), self.camera_direction,
                              tuple(self["subgoals"]))

    @property
    def subgoals(self):
        return self["subgoals"]
    
    def to_state_without_subgoals(self):
        return RobotState(tuple(self["pose"]),
                          self.camera_direction)
    @classmethod
    def from_state_without_subgoals(cls, robot_state, subgoals=tuple()):
        return RobotStateWithSubgoals(tuple(robot_state.pose),
                                      robot_state.camera_direction,
                                      subgoals=subgoals)

class SubgoalRewardModel(RewardModel):
    """This is a generic Subgoal reward model
    which gives 100 points if the robot accomplishes
    any new subgoal."""
    def __init__(self, ids, overlay=False):
        """
        Args:
            overlay (bool): If True, then the subgoal's reward is added on top
                            of the original RewardModel's reward. If False,
                            then only the subgoal's achievement will earn the
                            robot the reward.
        """
        super().__init__(ids)
        self._overlay = overlay
        
    def argmax(self, state, action, next_state, **kwargs):
        """
        argmax(self, state, action, next_state, **kwargs)
        Returns the most likely reward"""
        # Reward is 1 if picked up a target, -1 if wrong. -0.1 otherwise
        robot_id = self.ids["Robot"]
        robot_state = state.object_states[robot_id]
        next_robot_state = next_state.object_states[robot_id]        
        reward = -1

        cur_subgoals = {sg_name:sg_status for sg_name, sg_status in robot_state.subgoals}
        next_subgoals = {sg_name:sg_status for sg_name, sg_status in next_robot_state.subgoals}
        for goal_name in cur_subgoals:
            status = cur_subgoals[goal_name]
            next_status = next_subgoals[goal_name]
            if status == Subgoal.IP:
                if next_status == Subgoal.SUCCESS:
                    reward += 100.0
                elif next_status == Subgoal.FAIL:
                    reward -= 100.0
        if self._overlay:
            reward += super().argmax(state, action, next_state, **kwargs)
        return reward



class SubgoalPlanner(pomdp_py.Planner):
    """The subgoal planner works by replacing the robot state in
    the agent's belief by a state that has an additional attribute,
    "subgoals_achieved" which tracks which subgoals are achieved"""
    def __init__(self, ids, subgoals, planner):
        self.ids = ids
        self._robot_state_with_subgoals = None
        self._planner = planner
        self._subgoals = subgoals

    def update_subgoals(self, subgoals):
        self._subgoals = subgoals

    def plan(self, agent):
        print("Current subgoals: %s" % str(list(self._subgoals.keys())))
        robot_id = self.ids["Robot"]
        target_id = self.ids["Target"][0]
        robot_state = agent.belief.mpe().object_states[robot_id]
        self._robot_state_with_subgoals =\
            RobotStateWithSubgoals.from_state_without_subgoals(
                robot_state, subgoals=tuple((sg_name, Subgoal.IP)
                                            for sg_name in self._subgoals))
        # Create a temporary agent, with subgoal-aware transition/reward models
        belief = pomdp_py.OOBelief({
            robot_id:pomdp_py.Histogram({self._robot_state_with_subgoals.copy():1.0}),
            target_id:agent.belief.object_beliefs[target_id]})
        transition_model = oopomdp.OOTransitionModel(
            set(agent.transition_model.cond_effects)\
            | {(AchievingSubgoals(self.ids, self._subgoals),
                UpdateSubgoalStatus(self.ids))})
        reward_model = SubgoalRewardModel(self.ids)
        tmp_agent = pomdp_py.Agent(belief,
                                   agent.policy_model,
                                   transition_model,
                                   agent.observation_model,
                                   reward_model)
        if hasattr(agent, "tree"):
            tmp_agent.tree = agent.tree
        
        # Plan accordingly
        action = self._planner.plan(tmp_agent)
        
        # Record the subgoals achieved if execute this action; Note that the action
        # is not executed right now. We are just recording the subgoals
        next_mpe_state = transition_model.sample(tmp_agent.belief.mpe(), action)
        self._robot_state_with_subgoals = next_mpe_state.object_states[robot_id].copy()

        if isinstance(self._planner, pomdp_py.POUCT):
            agent.tree = tmp_agent.tree
        return action

    def update(self, agent, action, observation):
        # Now agent's belief has been updated
        robot_id = self.ids["Robot"]
        robot_state = agent.belief.mpe().object_states[robot_id]
        if self._robot_state_with_subgoals is not None:
            assert robot_state == self._robot_state_with_subgoals.to_state_without_subgoals(),\
                "After executing action, robot_state != robot_state_with_subgoals"
            
            # Check if any subgoal is achieved or failed. If so, call the trigger function
            new_subgoals = {}
            robot_state = self._robot_state_with_subgoals
            for subgoal_name, status in robot_state["subgoals"]:
                if status == Subgoal.SUCCESS:
                    next_subgoal = self._subgoals[subgoal_name].trigger_success(robot_state, action, observation)
                    if next_subgoal is not None:
                        new_subgoals[next_subgoal.name] = next_subgoal

                elif status == Subgoal.FAIL:
                    next_subgoal = self._subgoals[subgoal_name].trigger_fail(robot_state, action, observation)
                    if next_subgoal is not None:
                        new_subgoals[next_subgoal.name] = next_subgoal                        

                else:
                    new_subgoals[subgoal_name] = self._subgoals[subgoal_name]    
            self.update_subgoals(new_subgoals)
        self._planner.update(agent, action, observation)
        
    @property
    def last_num_sims(self):
        if isinstance(self._planner, pomdp_py.POUCT):
            return self._planner.last_num_sims
        else:
            return -1
