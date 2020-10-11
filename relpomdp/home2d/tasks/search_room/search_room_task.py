# Search room task. The robot is assumed to have access to
# room layout but not room types. If one wants to just
# reach a room then set the prior to be that room.

from relpomdp.home2d.domain.action import *
from relpomdp.home2d.domain.condition_effect import MoveEffect
from relpomdp.home2d.domain.env import Home2DEnvironment
from relpomdp.home2d.tasks.task import Task
from relpomdp.home2d.tasks.common.policy_model import *
from relpomdp.home2d.tasks.common.sensor import *
import relpomdp.oopomdp.framework as oopomdp
from relpomdp.home2d.domain.condition_effect import *
from relpomdp.home2d.domain.relation import *
from relpomdp.oopomdp.framework import Objstate, Objobs, OOObs
from relpomdp.home2d.planning.relations import RoomAttr


class Stop(Action):
    def __init__(self):
        super().__init__("stop")

class CanStop(oopomdp.Condition):
    """Pick up condition"""
    def satisfy(self, state, action):
        return isinstance(action, Stop)

class StopEffect(oopomdp.DeterministicTEffect):
    """assumes robot has access to room layout map"""
    def __init__(self, robot_id, room_type, grid_map):
        self.robot_id = robot_id
        self.room_type = room_type
        self.grid_map = grid_map
        
    def mpe(self, state, action, byproduct=None):
        next_state = state  # copy has already happened
        robot_state = next_state.object_states[self.robot_id]
        room_state = next_state.object_states[self.room_type]
        robot_room = self.grid_map.room_of(robot_state["pose"][:2])
        room_room = self.grid_map.room_of(room_state["pose"])
        if robot_room.name == room_room.name:
            room_state["reached"] = True
        return next_state
    
    
class RewardModel(pomdp_py.RewardModel):
    def __init__(self, robot_id, room_type):
        self.robot_id = robot_id
        self.room_type = room_type
    
    def sample(self, state, action, next_state, **kwargs):
        return self.argmax(state, action, next_state)
    
    def argmax(self, state, action, next_state, **kwargs):
        """
        argmax(self, state, action, next_state, **kwargs)
        Returns the most likely reward"""
        if isinstance(action, Stop):
            room_state = state.object_states[self.room_type]
            next_room_state = next_state.object_states[self.room_type]
            if next_room_state["reached"]:
                # if not room_state["reached"]:
                return 100.0
                # else:
                #     return -1.0
            else:
                return -100.0
        return -1.0

class CanObserve(oopomdp.Condition):
    """Condition to observe"""
    def satisfy(self, next_state, action):
        return True  # always can

class RoomObserveEffect(oopomdp.DeterministicOEffect):
    """If the robot state and the room state
    are in the same room then the robot gets
    the observation of the room type; Requires no
    access to room type;
    
    See comment below in mpe() regarding observation parts."""
    def __init__(self, robot_id, room_type, grid_map,
                 epsilon=1e-9, for_env=False):
        """knows_room_type (bool): True if robot knows room types. Default False.
        for_env (bool) True if this is used to simulate environment
            observation; that is the robot gets to see the room type."""
        self.robot_id = robot_id
        self.room_type = room_type
        self.grid_map = grid_map
        self.for_env = for_env
        super().__init__("sensing", epsilon=epsilon)  # ? not really a reason to name the type this way

    def probability(self, observation, next_state, action, byproduct=None):
        """
        observation: Observation actually received (should be the room
            type of where the robot is at). 
        """
        room_state = next_state.object_states[self.room_type]
        robot_state = next_state.object_states[self.robot_id]
        room_room = self.grid_map.room_of(room_state["pose"])
        robot_room = self.grid_map.room_of(robot_state["pose"][:2])

        if room_room == robot_room:
            expected_observation = self.mpe(next_state, action)
            if expected_observation == observation:
                return 1.0 - self.epsilon
            else:
                return self.epsilon
        else:
            # We will receive null observation (omitted) for other rooms
            # and we expect mpe() to generate such an observation too,
            # as implemented below 'objobs("null")'. Therefore, just return 1.0
            # Caveat though.. When we have observed the desired room type,
            # all other places become unnecessary; so marking them zero to
            # facilitate planning (i.e. stopping at the observed room)
            if observation.objclass == self.room_type:
                return self.epsilon
            else:
                return 1.0 - self.epsilon
        
    def mpe(self, next_state, action, byproduct=None):
        """Returns an OOState after applying this effect on `state`"""
        room_state = next_state.object_states[self.room_type]
        robot_state = next_state.object_states[self.robot_id]

        room_room = self.grid_map.room_of(room_state["pose"])
        robot_room = self.grid_map.room_of(robot_state["pose"][:2])

        # There are two parts of an observation:
        # - The first part is the type of room the robot is in now
        # - The second part is a null observation for "other areas";
        # The second part is omitted, because it will always be null.
        if self.for_env:
            # for environment, we can access the room type the robot is in
            return Objobs(robot_room.room_type)

        # Otherwise, the robot observes the room when in the same  room
        if room_room == robot_room:
            return Objobs(self.room_type)
        else:
            return Objobs("null")


class GreedyActionPrior(pomdp_py.ActionPrior):
    """greedy action prior for 'xy' motion scheme;
    Requires knowledge of grid map"""
    def __init__(self, robot_id, room_type, grid_map,
                 num_visits_init, val_init, motions={MoveE, MoveW, MoveN, MoveS}):
        self.robot_id = robot_id
        self.room_type = room_type
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

        room_state = state.object_states[self.room_type]
        if room_state["reached"] is False:
            # cur_dist = euclidean_dist(robot_state["pose"][:2], room_state["pose"])
            neighbors = {MoveEffect.move_by(robot_state["pose"][:2], action.motion):action
                         for action in self.legal_motions[robot_state["pose"][:2]]}
            for next_robot_pose in neighbors:
                # Prefer action to move into the room where
                # the sampled target state is located
                action = neighbors[next_robot_pose]
                next_room = self.grid_map.room_of(next_robot_pose[:2])
                object_room = self.grid_map.room_of(room_state["pose"])
                if object_room == next_room:
                   # and euclidean_dist(next_robot_pose, room_state["pose"]) < cur_dist:
                    preferences.add((action,
                                     self.num_visits_init, self.val_init))
        return preferences

        

class SearchRoomTask(Task):
    """
    Search room task: Searches for a room with a given type
    """
    def __init__(self,
                 robot_id,
                 room_type,
                 grid_map):
        self.robot_id = robot_id
        self.room_type = room_type
        motions = {MoveN, MoveS, MoveE, MoveW}

        cond_effects_t = []
        legal_motions = grid_map.compute_legal_motions(motions)
        cond_effects_t.append((CanMove(robot_id, legal_motions), MoveEffect(robot_id)))
        cond_effects_t.append((CanStop(), StopEffect(robot_id, room_type, grid_map)))
        transition_model = oopomdp.OOTransitionModel(cond_effects_t)

        cond_effects_o = [(CanObserve(),
                           RoomObserveEffect(robot_id, room_type, epsilon=1e-12,
                                             grid_map=grid_map))]
        observation_model = oopomdp.OOObservationModel(cond_effects_o)

        reward_model = RewardModel(robot_id, room_type)

        if grid_map is not None:
            action_prior = GreedyActionPrior(robot_id, room_type, grid_map, 10, 100)
            policy_model = PreferredPolicyModel(action_prior,
                                                other_actions={Stop()})
        else:
            policy_model = PolicyModel(robot_id, motions=motions,
                                       other_actions={Stop()},
                                       grid_map=grid_map)
        
        super().__init__(transition_model,
                         observation_model,
                         reward_model,
                         policy_model)

    def __str__(self):
        return "SearchRoomTask(%s)" % self.room_type

    def __repr__(self):
        return str(self)

    # Functions used during planning
    def is_done(self, env, action):
        """Check if the task is done given environment state
        and the last action taken by agent. """
        return action == Stop() and\
           env.grid_map.room_of(env.robot_state["pose"][:2]).room_type == self.room_type

    def get_result(self, agent, grid_map):
        """Get result in the form of an object observation"""
        # The observation is that the type of room appears in the room id,
        # as believed by the agent.        
        room_state = agent.belief.object_beliefs[self.room_type].mpe()
        return objobs(self.room_type, 
                      room_id=grid_map.room_of(room_state["pose"]).name)  # name === room_id

    def get_prior(self, grid_map, prior_type="uniform", **kwargs):
        room_hist = {}
        total_prob = 0
        for x in range(grid_map.width):
            for y in range(grid_map.length):
                state = Objstate(self.room_type, pose=(x,y), reached=False)
                if prior_type == "uniform":
                    room_hist[state] = 1.0
                elif prior_type == "informed":
                    if grid_map.room_of((x,y)).room_type != self.room_type:
                        room_hist[state] = 0.0
                    else:
                        room_hist[state] = 1.0
                total_prob += room_hist[state]
        # Normalize
        for state in room_hist:
            room_hist[state] /= total_prob

        # Return OOBelief, or just histogram
        if "robot_state" in kwargs:
            robot_state = kwargs["robot_state"]
            return pomdp_py.OOBelief({self.room_type: pomdp_py.Histogram(room_hist),
                                      self.robot_id: pomdp_py.Histogram({robot_state:1.0})})
        else:
            return pomdp_py.Histogram(room_hist)

    def get_env(self, global_env=None, **kwargs):
        """
        Returns a Home2DEnvironment for this task (with suitable initial state)
        given a global task environment. If global task environment
        is not provided, then there needs to be appropriate arguments in kwargs
        """
        if global_env is None:
            robot_state = kwargs.get("robot_state", None)
            grid_map = kwargs.get("grid_map", None)
        else:
            robot_state = global_env.robot_state
            grid_map = global_env.grid_map

        room = None
        for room_name in grid_map.rooms:
            if grid_map.rooms[room_name].room_type == self.room_type:
                room = grid_map.rooms[room_name]
                break
        init_state = {self.robot_id: robot_state,
                      self.room_type: Objstate(self.room_type,
                                          pose=room.center_of_mass, reached=False)}
        env = Home2DEnvironment(self.robot_id,
                                grid_map, init_state,
                                reward_model=self.reward_model)
        env.transition_model.cond_effects.append(
            (CanStop(), StopEffect(self.robot_id, self.room_type, grid_map))
        )
        cond_effects_o = [(CanObserve(),
                           RoomObserveEffect(self.robot_id, self.room_type, epsilon=1e-12,
                                             grid_map=grid_map,
                                             for_env=True))]
        env.observation_model = oopomdp.OOObservationModel(cond_effects_o)
        return env
        
        
    def step(self, env, agent, planner):
        """
        The agent is assumed to be using an OOBelief
        """        
        action = planner.plan(agent)
        for a in agent.tree.children:
            print(a, agent.tree.children[a].value)
        
        reward = env.state_transition(action, execute=True)
        observation = env.observation_model.sample(env.state, action)

        # Belief update
        robot_state = env.state.object_states[self.robot_id]
        room_types = set(env.grid_map.rooms[r].room_type
                         for r in env.grid_map.rooms)
        # Create a next state space on top of new robot pose, and all room types.
        cur_belief = pomdp_py.Histogram({
            oopomdp.OOState({self.room_type: room_state,
                             self.robot_id: robot_state}) : agent.belief.object_beliefs[self.room_type][room_state]
            for room_state in agent.belief.object_beliefs[self.room_type]})        
        new_belief = pomdp_py.update_histogram_belief(cur_belief,
                                                      action, observation,
                                                      agent.observation_model,
                                                      agent.transition_model,
                                                      static_transition=True)
        # Take just the room state from this
        new_belief = pomdp_py.Histogram({state.object_states[self.room_type]:
                                         new_belief[state]
                                         for state in new_belief})
        agent.belief.set_object_belief(self.room_type, new_belief)        
        agent.belief.set_object_belief(self.robot_id, pomdp_py.Histogram({robot_state:1.0}))
        observation_planner = agent.observation_model.sample(env.state, action)
        planner.update(agent, action, observation_planner)
        return action, observation, reward
