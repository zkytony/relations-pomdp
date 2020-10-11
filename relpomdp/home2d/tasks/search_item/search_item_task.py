
# In this task, the agent searches for one item,
# and picks it up.

from relpomdp.home2d.domain.action import *
from relpomdp.home2d.domain.env import Home2DEnvironment
from relpomdp.home2d.tasks.task import Task
from relpomdp.home2d.tasks.common.policy_model import PolicyModel, PreferredPolicyModel
from relpomdp.home2d.tasks.common.sensor import *
import relpomdp.oopomdp.framework as oopomdp
from relpomdp.oopomdp.framework import Objstate, Objobs, OOObs
from relpomdp.home2d.domain.condition_effect import *
from relpomdp.home2d.domain.relation import *
import pomdp_py

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
                    observation = Objobs(objstate.objclass, pose=objstate["pose"])
                    obs[objid] = observation
        return OOObs(obs)


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


class GreedyActionPrior(pomdp_py.ActionPrior):
    """greedy action prior for 'xy' motion scheme;
    Requires knowledge of grid map"""
    def __init__(self, robot_id, target_id, grid_map,
                 num_visits_init, val_init, motions={MoveE, MoveW, MoveN, MoveS}):
        self.robot_id = robot_id
        self.target_id = target_id
        self.grid_map = grid_map
        self.legal_motions = grid_map.compute_legal_motions(motions)
        self.motions = motions
        self.num_visits_init = num_visits_init
        self.val_init = val_init

    def get_preferred_actions(self, state, history):
        """Get preferred actions. This can be used by a rollout policy as well."""
        robot_state = state.object_states[self.robot_id]
        cur_room = self.grid_map.room_of(robot_state["pose"][:2])
        preferences = set()

        target_state = state.object_states[self.target_id]
        if target_state["is_found"] is False:
            cur_dist = euclidean_dist(robot_state["pose"][:2], target_state["pose"])
            neighbors = {MoveEffect.move_by(robot_state["pose"][:2], action.motion):action
                         for action in self.legal_motions[robot_state["pose"][:2]]}
            for next_robot_pose in neighbors:
                # Prefer action to move into the room where
                # the sampled target state is located
                action = neighbors[next_robot_pose]
                next_room = self.grid_map.room_of(next_robot_pose[:2])
                object_room = self.grid_map.room_of(target_state["pose"])
                if object_room == next_room\
                   and euclidean_dist(next_robot_pose, target_state["pose"]) < cur_dist:
                    preferences.add((action,
                                     self.num_visits_init, self.val_init))
        return preferences
    
    

class SearchItemTask(Task):

    """
    In this task, the agent searches for one item,
    and picks it up.
    """

    def __init__(self,
                 robot_id,
                 target_id,
                 target_class,
                 sensor,
                 grid_map=None,  # If None, then the agent does not know the map at all
                 within_room=False):  # If True, searches only within the current room
        self.robot_id = robot_id
        self.target_id = target_id
        self.target_class = target_class
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

        if grid_map is not None:
            # Greedy policy
            action_prior = GreedyActionPrior(robot_id, target_id, grid_map, 10, 100)
            policy_model = PreferredPolicyModel(action_prior,
                                                other_actions={Pickup()})
        else:
            # Random rollout
            policy_model = PolicyModel(robot_id, motions=motions,
                                       other_actions={Pickup()},
                                       grid_map=grid_map)
        reward_model = RewardModel(robot_id, target_id, grid_map=grid_map, within_room=within_room)
        super().__init__(transition_model,
                         observation_model,
                         reward_model,
                         policy_model)



    def is_done(self, env, *args):
        return env.state.object_states[self.target_id]["is_found"]

    def get_prior(self, grid_map, prior_type="uniform", env=None, **kwargs):
        target_hist = {}
        total_prob = 0
        for x in range(grid_map.width):
            for y in range(grid_map.length):
                state = Objstate(self.target_class, pose=(x,y), is_found=False)
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

        # Return OOBelief, or just histogram            
        if "robot_state" in kwargs:
            robot_state = kwargs["robot_state"]
            return pomdp_py.OOBelief({self.target_id: pomdp_py.Histogram(target_hist),
                                      self.robot_id: pomdp_py.Histogram({robot_state:1.0})})
        else:
            return pomdp_py.Histogram(target_hist)            

    def step(self, env, agent, planner):
        """
        The agent is assumed to be using an OOBelief
        """
        action = planner.plan(agent)
        reward = env.state_transition(action, execute=True)
        observation = agent.observation_model.sample(env.state, action)

        # Belief update
        robot_state = env.state.object_states[self.robot_id]
    
        cur_belief = pomdp_py.Histogram({
            oopomdp.OOState({self.target_id: target_state,
                             self.robot_id: robot_state}) : agent.belief.object_beliefs[self.target_id][target_state]
            for target_state in agent.belief.object_beliefs[self.target_id]})
        new_belief = pomdp_py.update_histogram_belief(cur_belief,
                                                      action, observation,
                                                      agent.observation_model,
                                                      agent.transition_model,
                                                      static_transition=True)
        # Take just the target state from this
        new_belief = pomdp_py.Histogram({state.object_states[self.target_id]:
                                         new_belief[state]
                                         for state in new_belief})
        agent.belief.set_object_belief(self.target_id, new_belief)
        agent.belief.set_object_belief(self.robot_id, pomdp_py.Histogram({robot_state:1.0}))
        planner.update(agent, action, observation)
        return action, observation, reward

    def get_env(self, global_env=None, **kwargs):
        """
        Returns a Home2DEnvironment for this task (with suitable initial state)
        given a global task environment. If global task environment
        is not provided, then there needs to be appropriate arguments in kwargs
        """
        if global_env is None:
            init_state = kwargs.get("init_state", None) # dict from id to state
            grid_map = kwargs.get("grid_map", None) # dict from id to state
        else:
            init_state = env.state
            grid_map = env.grid_map
        env = Home2DEnvironment(self.robot_id,
                                grid_map, init_state,
                                reward_model=self.reward_model)
        return env

# Unittest
def unittest():
    from relpomdp.home2d.domain.maps import all_maps
    grid_map = all_maps["map_small_0"]()

    # Building object state
    robot_id = 1
    robot_state = Objstate("Robot",
                           pose=(0,0,0),
                           camera_direction="+x")
    salt_id = 10
    salt_state = Objstate("Salt",
                          pose=(3,3))

    pepper_id = 15
    pepper_state = Objstate("Pepper",
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
            state = Objstate(target_class, pose=(x,y), is_found=False)
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
