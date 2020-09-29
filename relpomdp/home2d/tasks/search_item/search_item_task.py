
# In this task, the agent searches for one item,
# and picks it up.

from relpomdp.home2d.domain.action import *
from relpomdp.home2d.domain.env import Home2DEnvironment
from relpomdp.home2d.tasks.task import Task
from relpomdp.home2d.tasks.common.policy_model import PolicyModel
from relpomdp.home2d.tasks.common.sensor import *
import relpomdp.oopomdp.framework as oopomdp
from relpomdp.home2d.domain.condition_effect import *
from relpomdp.home2d.domain.relation import *
from relpomdp.home2d.utils import objstate, objobs, ooobs

class Pickup(Action):
    """Pick up action"""
    def __init__(self):
        super().__init__("pickup")

class CanPickup(oopomdp.Condition):
    """Pick up condition"""
    def __init__(self, robot_id, target_id):
        self.robot_id = robot_id
        self.target_id = target_id
        
    def satisfy(self, state, action):
        if not isinstance(action, Pickup):
            return False
        robot_state = state.object_states[self.robot_id]
        target_state = state.object_states[self.target_id]
        if is_on(robot_state, target_state)\
           and not target_state["is_found"]:
            return True, self.target_id
        return False
            
class PickupEffect(oopomdp.DeterministicTEffect):
    """Pick up effect: Deterministically pick up"""
    def __init__(self):
        super().__init__("pickup")
        
    def mpe(self, state, action, picking_objid):
        """Returns an OOState after applying this effect on `state`"""
        next_state = state
        next_state.object_states[picking_objid]["is_found"] = True
        return next_state


# Observation condition / effects
class CanObserve(oopomdp.Condition):
    """Condition to observe"""
    def satisfy(self, next_state, action):
        return True  # always can

class ObjectObserveEffect(oopomdp.DeterministicOEffect):
    """Effect of observation"""
    def __init__(self, sensor, robot_id, epsilon=1e-9):
        self.robot_id = robot_id
        self.sensor = sensor
        super().__init__("sensing", epsilon=epsilon)  # ? not really a reason to name the type this way

    def probability(self, observation, next_state, action, byproduct=None):
        """Returns the probability of getting `observation` if applying
        this effect on `state` given `action`."""
        expected_observation = self.mpe(next_state, action)
        modeled_objs = [objid for objid in next_state.object_states\
                        if "pose" in next_state.object_states[objid].attributes\
                        and objid != self.robot_id]
        if expected_observation == observation.for_objs(modeled_objs):
            return 1.0 - self.epsilon
        else:
            return self.epsilon        
        
    def mpe(self, next_state, action, byproduct=None):
        """Returns an OOState after applying this effect on `state`"""
        robot_state = next_state.object_states[self.robot_id]
        obs = {}
        for objid in next_state.object_states:
            objstate = next_state.object_states[objid]
            if "pose" in objstate.attributes and objid != self.robot_id:
                if self.sensor.within_range(robot_state["pose"], objstate["pose"]):
                    observation = objobs(objstate.objclass, pose=objstate["pose"])
                    obs[objid] = observation
        return ooobs(obs)


class RewardModel(pomdp_py.RewardModel):
    """
    Reward model for search item task
    """
    def __init__(self, robot_id, target_id, grid_map=None, within_room=False):
        self.robot_id = robot_id
        self.target_id = target_id
        self.grid_map = grid_map
        self.within_room = within_room
    
    def sample(self, state, action, next_state, **kwargs):
        return self.argmax(state, action, next_state)
    
    def argmax(self, state, action, next_state, **kwargs):
        """
        argmax(self, state, action, next_state, **kwargs)
        Returns the most likely reward"""
        # Reward is 100 if picked up a target, -100 if wrong. -1 otherwise
        if self.within_room:
            cur_room = self.grid_map.room_of(state.object_states[self.robot_id]["pose"][:2])
            next_room = self.grid_map.room_of(next_state.object_states[self.robot_id]["pose"][:2])
            if next_room != cur_room:
                return -100.0
                    
        if isinstance(action, Pickup):
            found = state.object_states[self.target_id]["is_found"]
            next_found = next_state.object_states[self.target_id]["is_found"]
            if next_found:
                if not found: 
                    return 100.0
                else:
                    return -1.0
            else:
                return -100.0
        return -1.0
    

class SearchItemTask(Task):

    """
    In this task, the agent searches for one item,
    and picks it up.
    """

    def __init__(self,
                 robot_id,
                 target_id,
                 sensor,
                 grid_map=None,  # If None, then the agent does not know the map at all
                 within_room=False):  # If True, searches only within the current room
        self.robot_id = robot_id
        self.target_id = target_id
        motions = {MoveN, MoveS, MoveE, MoveW}

        cond_effects_t = []
        if grid_map is None:
            cond_effects_t.append((CanMove(robot_id, None), MoveEffect(robot_id)))
        else:
            legal_motions = grid_map.compute_legal_motions(motions)
            cond_effects_t.append((CanMove(robot_id, legal_motions), MoveEffect(robot_id)))
        cond_effects_t.append((CanPickup(robot_id, target_id), PickupEffect()))
        transition_model = oopomdp.OOTransitionModel(cond_effects_t)

        cond_effects_o = [(CanObserve(), ObjectObserveEffect(sensor, robot_id, epsilon=1e-12))]
        observation_model = oopomdp.OOObservationModel(cond_effects_o)
        policy_model = PolicyModel(robot_id, motions=motions,
                                   other_actions={Pickup()},
                                   grid_map=grid_map)
        reward_model = RewardModel(robot_id, target_id, grid_map=grid_map, within_room=within_room)
        super().__init__(transition_model,
                         observation_model,
                         reward_model,
                         policy_model)


# Unittest
def unittest():
    from relpomdp.home2d.domain.maps import all_maps
    grid_map = all_maps["map_small_0"]()

    # Building object state
    robot_id = 1
    robot_state = objstate("Robot",
                           pose=(0,0,0),
                           camera_direction="+x")
    salt_id = 10
    salt_state = objstate("Salt",
                          pose=(3,3))

    pepper_id = 15
    pepper_state = objstate("Pepper",
                            pose=(3,2))
    
    init_state = {robot_id: robot_state,
                  salt_id: salt_state,
                  pepper_id: pepper_state}
    env = Home2DEnvironment(robot_id,
        grid_map, init_state)

    target_id = salt_id
    target_class = "Salt"
    salt_state["is_found"] = False

    # Obtain prior
    prior_type = "uniform"
    target_hist = {}
    total_prob = 0
    for x in range(grid_map.width):
        for y in range(grid_map.length):
            state = objstate(target_class, pose=(x,y), is_found=False)
            if prior_type == "uniform":
                target_hist[state] = 1.0
            elif prior_type == "informed":
                if (x,y) != env.state.object_states[target_id]["pose"]:
                    target_hist[state] = 0.0
                else:
                    target_hist[state] = 1.0
            total_prob += target_hist[state]
    # Normalize
    for state in target_hist:
        target_hist[state] /= total_prob

    init_belief = pomdp_py.OOBelief({target_id: pomdp_py.Histogram(target_hist),
                                     robot_id: pomdp_py.Histogram({env.robot_state:1.0})})    

    sensor = Laser2DSensor(robot_id, env.grid_map,  # the sensor uses the grid map for wall blocking
                           fov=90, min_range=1, max_range=2,
                           angle_increment=0.5)
    task = SearchItemTask(robot_id, salt_id, sensor)    
    agent = task.to_agent(init_belief)

    # Create planner and make a plan
    planner = pomdp_py.POUCT(max_depth=10,
                             discount_factor=0.95,
                             num_sims=100,
                             exploration_const=100,
                             rollout_policy=agent.policy_model)
    print(planner.plan(agent))


if __name__ == "__main__":
    unittest()    
