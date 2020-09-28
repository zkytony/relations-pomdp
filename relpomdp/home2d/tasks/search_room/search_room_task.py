# Search room task.
# The robot either has access to the full grid map
# (in which case the problem is essentially an MDP),
# a grid map with only room layout, or no grid map.
# In the first two cases, the robot gets to observe
# the room ID it is in. In the first case, the robot
# gets to additionally observe the room type.



from relpomdp.home2d.domain.action import *
from relpomdp.home2d.domain.condition_effect import MoveEffect
from relpomdp.home2d.domain.env import Home2DEnvironment
from relpomdp.home2d.tasks.task import Task
from relpomdp.home2d.tasks.common.policy_model import PolicyModel
from relpomdp.home2d.tasks.common.sensor import *
import relpomdp.oopomdp.framework as oopomdp
from relpomdp.home2d.domain.condition_effect import *
from relpomdp.home2d.domain.relation import *
from relpomdp.home2d.utils import objstate, objobs, ooobs

class Stop(Action):
    def __init__(self):
        super().__init__("stop")
    
class RewardModel(pomdp_py.RewardModel):
    def __init__(self, robot_id, room_type,
                 grid_map=None, knows_room_type=False):
        self.robot_id = robot_id
        self.room_type = room_type
        self.grid_map = grid_map
        self.knows_room_type = knows_room_type
    
    def sample(self, state, action, next_state, **kwargs):
        return self.argmax(state, action, next_state)
    
    def argmax(self, state, action, next_state, **kwargs):
        """
        argmax(self, state, action, next_state, **kwargs)
        Returns the most likely reward"""
        if isinstance(action, Stop):
            robot_state = state.object_states[self.robot_id]
            next_robot_state = next_state.object_states[self.robot_id]
            if self.grid_map is None or self.knows_room_type is False:
                # The robot has no access to grid map or doesn't
                # know room types; So it can only check based on the state
                if next_robot_state["room_type"] == self.room_type:
                    return 100.0
                else:
                    return -100.0
            else:
                # The robot does have access to the grid map
                cur_room = self.grid_map.room_of(robot_state["pose"][:2])
                next_room = self.grid_map.room_of(next_robot_state["pose"][:2])
                if next_room.room_type == self.room_type:
                    return 100.0
                else:
                    return -100.0
        return -1

class CanObserve(oopomdp.Condition):
    """Condition to observe"""
    def satisfy(self, next_state, action):
        return True  # always can

class RoomObserveEffect(oopomdp.DeterministicOEffect):
    def __init__(self, robot_id, epsilon=1e-9):
        self.robot_id = robot_id
        super().__init__("sensing", epsilon=epsilon)  # ? not really a reason to name the type this way
        
    def mpe(self, next_state, action, byproduct=None):
        """Returns an OOState after applying this effect on `state`"""
        robot_state = next_state.object_states[self.robot_id]
        return objobs("room", room_type=robot_state["room_type"])

class MoveEffectRoom(oopomdp.DeterministicOEffect):
    """Should only be used in the case where the robot knows the room type;
    Must happen after MoveEffect."""
    def __init__(self, robot_id, grid_map, epsilon=1e-12):
        self.robot_id = robot_id
        self.grid_map = grid_map
        super().__init__("move", epsilon=epsilon)        
        
    def mpe(self, state, action, byproduct=None):
        next_state = state  # copy has already happened
        next_room_type = self.grid_map.room_of(
            state.object_states[self.robot_id]["pose"][:2]).room_type
        next_state.object_states[self.robot_id]["room_type"] = next_room_type
        return next_state

class SearchRoomTask(Task):
    """
    Search room task: Searches for a room with a given type
    """
    def __init__(self,
                 robot_id,
                 room_type,
                 grid_map=None,
                 knows_room_type=False):
        self.robot_id = robot_id
        self.room_type = room_type
        motions = {MoveN, MoveS, MoveE, MoveW}

        cond_effects_t = []
        if grid_map is None:
            cond_effects_t.append((CanMove(robot_id, None), MoveEffect(robot_id)))
        else:
            legal_motions = grid_map.compute_legal_motions(motions)
            cond_effects_t.append((CanMove(robot_id, legal_motions), MoveEffect(robot_id)))
            if knows_room_type:
                cond_effects_t.append((CanMove(robot_id, legal_motions), MoveEffectRoom(robot_id, grid_map)))
            
        transition_model = oopomdp.OOTransitionModel(cond_effects_t)

        cond_effects_o = [(CanObserve(), RoomObserveEffect(robot_id, epsilon=1e-12))]
        observation_model = oopomdp.OOObservationModel(cond_effects_o)

        reward_model = RewardModel(robot_id, room_type,
                                   grid_map=grid_map, knows_room_type=knows_room_type)
        policy_model = PolicyModel(robot_id, motions=motions,
                                   other_actions={Stop()},
                                   grid_map=grid_map)
        
        super().__init__(transition_model,
                         observation_model,
                         reward_model,
                         policy_model)
