import pomdp_py
import copy
from relpomdp.oopomdp.abstraction import AbstractAttribute
from relpomdp.object_search.world_specs.build_world import small_map1
from relpomdp.object_search.state import *
from relpomdp.object_search.agent import *
from relpomdp.object_search.env import *
from relpomdp.object_search.sensor import *
from relpomdp.object_search.reward_model import RewardModel
from relpomdp.object_search.tests.worlds import *
from relpomdp.object_search.world_specs.build_world import *
import relpomdp.oopomdp.framework as oopomdp
import time
# TODO: Make this 


class RoomAttr(AbstractAttribute):
    """This is an abstract attribute for Pose"""
    def __init__(self, room_name):
        super().__init__("room", room_name)
    @property
    def room_name(self):
        return self.value
    def copy(self):
        return RoomAttr(self.value)
    def reverse_image(self, grid_map):
        # Returns a set of Pose attributes
        room = grid_map.rooms[self.room_name]
        return [Pose(loc) for loc in room.locations]
    @classmethod
    def abstractify(self, pose_attribute, grid_map):
        if type(pose_attribute) == tuple:
            room_name = grid_map.room_of(pose_attribute[:2])
        else:  # Pose class
            room_name = grid_map.room_of(pose_attribute.value[:2])
        return RoomAttr(room_name)

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
    def achieve(self, state, action):
        pass
    def fail(self, state, action):
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
    

class ReachRoomSubgoal(Subgoal):
    """
    This subgoal is achieved when the robot reaches a particular room.
    """
    def __init__(self, ids, room_type, grid_map):
        self.ids = ids
        self.grid_map = grid_map
        self.room_type = room_type
        super().__init__("Reach-%s" % room_type)
        
    def achieve(self, state, action):
        # Achieves the goal when the robot is at the center of mass of the room
        robot_id = self.ids["Robot"]
        robot_state = state.object_states[robot_id]
        room_attr = RoomAttr.abstractify(robot_state.pose, self.grid_map)
        room = self.grid_map.rooms[room_attr.room_name]
        return room.room_type == self.room_type\
            and robot_state.pose[:2] == room.center_of_mass

    def fail(self, state, action):
        return isinstance(action, Pickup)

    def trigger_success(self, robot_state, action, observation):
        room_attr = RoomAttr.abstractify(robot_state.pose, self.grid_map)
        subgoal = SearchRoomSubgoal(self.ids, room_attr.room_name, self.grid_map)
        return subgoal
            

class SearchRoomSubgoal(Subgoal):
    """
    This subgoal is achieved when the target object is found
    within the room being searched. It fails when either the
    robot steps outside of the room or 
    """
    def __init__(self, ids, room_name, grid_map):
        self.ids = ids
        self.room_name = room_name
        self.grid_map = grid_map
        super().__init__("Search-%s" % room_name)

    def achieve(self, state, action):
        robot_id = self.ids["Robot"]
        robot_state = state.object_states[robot_id]
        room_attr = RoomAttr.abstractify(robot_state.pose, self.grid_map)
        if room_attr.room_name != self.room_name:
            return False
        for objid in self.ids["Target"]:
            if state.object_states[objid]["is_found"]:
                return True
        return False

    def fail(self, state, action):
        robot_id = self.ids["Robot"]        
        robot_state = state.object_states[robot_id]
        if isinstance(action, Pickup):
            for objid in self.ids["Target"]:
                objstate = state.object_states[objid]
                if not (objstate.pose == robot_state.pose[:2]\
                        and not objstate["is_found"]):
                    return True
        return False

def interpret_subgoal(string, **kwargs):
    ids = kwargs.get("ids", None)
    grid_map = kwargs.get("grid_map", None)
    if string.startswith("Reach"):
        room_type = string.split("_")[1]
        subgoal = ReachRoomSubgoal(ids, room_type, grid_map)
    else:
        room_name = string.split("_")[1]
        subgoal = SearchRoomSubgoal(ids, room_name, grid_map)
    return subgoal
        
    
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
        robot_id = self.ids["Robot"]
        robot_state = state.object_states[robot_id]
        start = time.time()
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
    


if __name__ == "__main__":
    # Test
    ids, grid_map, init_state, colors = salt_pepper_1()

    # Obtain prior
    target_hist = {}
    total_prob = 0
    for x in range(grid_map.width):
        for y in range(grid_map.length):
            state = ItemState("Salt", (x,y))
            target_hist[state] = 1.0
            total_prob += target_hist[state]
    # Normalize
    for state in target_hist:
        target_hist[state] /= total_prob
    
    robot_id = ids["Robot"]
    target_id = ids["Target"][0]    
    env = ObjectSearchEnvironment(ids,
                                  grid_map,
                                  init_state)
    sensor = Laser2DSensor(robot_id, env.grid_map, fov=90, min_range=1, max_range=2,
                           angle_increment=0.5)
    # Build a regular agent
    init_belief = oopomdp.OOBelief({target_id: pomdp_py.Histogram(target_hist),
                                     robot_id: pomdp_py.Histogram({env.robot_state:1.0})},
                                   oo_state_class=JointState)
    agent = ObjectSearchAgent(env.grid_map, sensor, env.ids,
                              init_belief)

    # Build the agent used for subgoal planning
    sg = ReachRoomSubgoal(ids, "Kitchen", grid_map)
    subgoals = {sg.name: sg}
    robot_id = ids["Robot"]
    robot_state = agent.belief.mpe().object_states[robot_id]    
    robot_state_with_subgoals =\
        RobotStateWithSubgoals.from_state_without_subgoals(
            robot_state, subgoals=tuple((sg_name, Subgoal.IP)
                                        for sg_name in subgoals))
    belief = oopomdp.OOBelief({robot_id:pomdp_py.Histogram({robot_state_with_subgoals.copy():1.0}),
                               target_id:agent.belief.object_beliefs[target_id]},
                              oo_state_class=JointState) 
    transition_model = oopomdp.OOTransitionModel(
        set(agent.transition_model.cond_effects)\
        | {(AchievingSubgoals(ids, subgoals),
            UpdateSubgoalStatus(ids))})
    reward_model = SubgoalRewardModel(ids)
    tmp_agent = pomdp_py.Agent(belief,
                               agent.policy_model,
                               transition_model,
                               agent.observation_model,
                               reward_model)

    start = time.time()
    for i in range(100):
        next_state, observation, reward, _ = pomdp_py.sample_generative_model(
            tmp_agent, tmp_agent.belief.mpe(), MoveW)
    total_time = time.time() - start
    print("Sampling generative model 100 times used %.9fs" % total_time)
    print(next_state)
