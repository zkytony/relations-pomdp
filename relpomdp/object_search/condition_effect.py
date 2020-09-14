import relpomdp.oopomdp.framework as oopomdp
from relpomdp.object_search.relation import *
from relpomdp.object_search.action import *
from relpomdp.object_search.state import *
from relpomdp.object_search.observation import *

# Conditions and effects
class CanMove(oopomdp.Condition):
    def __init__(self, ids, motion_policy):
        self.motion_policy = motion_policy
        self.ids = ids  # map from object class to id or a set of ids
    
    # Strictly speaking OO-MDP expects conditions like CanMoveE, CanMoveN, etc.
    # but that is too verbose
    def satisfy(self, state, action):
        if not isinstance(action, Move):
            return False
        robot_state = state.object_states[self.ids["Robot"]]
        return action in self.motion_policy.valid_motions(robot_state.pose)
    
class MoveEffect(oopomdp.DeterministicTEffect):
    """Deterministically move"""
    def __init__(self, ids):
        self.ids = ids
        super().__init__("arithmetic")  # ? not really a reason to name the type this way

    @classmethod
    def move_by(self, robot_pose, motion):
        dx, dy, th = motion
        if isinstance(robot_pose, Pose):
            rx, ry = robot_pose.value[:2]
        else:
            rx, ry = robot_pose[:2]
        return Pose((rx + dx, ry + dy, th))
        
    def mpe(self, state, action, byproduct=None):
        """Returns an OOState after applying this effect on `state`"""
        robot_state = state.object_states[self.ids["Robot"]]
        next_state = state.copy()
        next_robot_state = next_state.object_states[self.ids["Robot"]]
        next_robot_state["pose"] = MoveEffect.move_by(robot_state.pose, action.motion)
        next_robot_state["camera_direction"] = action.name
        return next_state

class CanPickup(oopomdp.Condition):
    def __init__(self, ids):
        self.ids = ids
        
    def satisfy(self, state, action):
        if not isinstance(action, Pickup):
            return False
        robot_state = state.object_states[self.ids["Robot"]]
        for objid in self.ids["Target"]:
            if is_on(robot_state, state.object_states[objid])\
               and not state.object_states[objid].is_found:
                return True, objid
        return False
            
class PickupEffect(oopomdp.DeterministicTEffect):
    """Deterministically move"""
    def __init__(self):
        super().__init__("constant")  # ? not really a reason to name the type this way
        
    def mpe(self, state, action, picking_objid):
        """Returns an OOState after applying this effect on `state`"""
        next_state = state
        next_state.object_states[picking_objid]["is_found"] = True
        return next_state

# Observation condition / effects
class CanObserve(oopomdp.Condition):
    def __init__(self, ids):
        self.ids = ids
        
    def satisfy(self, next_state, action):
        return True  # always can

class ObjectObserveEffect(oopomdp.DeterministicOEffect):
    def __init__(self, sensor, ids, epsilon=1e-9):
        self.ids = ids
        self.sensor = sensor
        super().__init__("sensing", epsilon=epsilon)  # ? not really a reason to name the type this way

    def probability(self, observation, next_state, action, byproduct=None):
        """Returns the probability of getting `observation` if applying
        this effect on `state` given `action`."""
        expected_observation = self.mpe(next_state, action)
        modeled_objs = [objid for objid in next_state.object_states\
                        if isinstance(next_state.object_states[objid], PoseState)]
        if expected_observation == observation.for_objs(modeled_objs):
            return 1.0 - self.epsilon
        else:
            return self.epsilon        
        
    def mpe(self, next_state, action, byproduct=None):
        """Returns an OOState after applying this effect on `state`"""
        robot_state = next_state.object_states[self.ids["Robot"]]
        obs = {}
        for objid in next_state.object_states:
            objstate = next_state.object_states[objid]
            if isinstance(objstate, PoseState):
                if self.sensor.within_range(robot_state.pose, objstate.pose):
                    observation = ItemObservation(objstate.objclass, objstate.pose)
                    obs[objid] = observation
        return JointObservation(obs)

class RoomObserveEffect(oopomdp.DeterministicOEffect):
    def __init__(self, sensor, ids, grid_map, epsilon=1e-9):
        self.ids = ids
        self.sensor = sensor
        self.grid_map = grid_map
        super().__init__("sensing", epsilon=epsilon)  # ? not really a reason to name the type this way
        
    def mpe(self, next_state, action, byproduct=None):
        """Returns an OOState after applying this effect on `state`"""
        # If the robot enters a room, it observes the type of that room
        # (unrealistic assumption)
        robot_state = next_state.object_states[self.ids["Robot"]]
        room_name = self.grid_map.room_of(robot_state.pose[:2])
        return RoomObservation(room_name)
