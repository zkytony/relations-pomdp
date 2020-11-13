# nk_agent: Agent without prior knowledge of the map

import pomdp_py
from pomdp_py import Histogram
from relpomdp.home2d.domain.maps.grid_map import GridMap
from relpomdp.home2d.domain.action import *
from relpomdp.home2d.agent.observation_model import CanObserve, ObserveEffect
from relpomdp.home2d.agent.transition_model import CanDeclareFound, DeclareFoundEffect, DeclareFound
from relpomdp.home2d.agent.reward_model import DeclareFoundRewardModel
from relpomdp.home2d.agent.sensor import Laser2DSensor, SensorCache
from relpomdp.home2d.agent.partial_map import PartialGridMap
from relpomdp.home2d.domain.condition_effect import CanMove, MoveEffect
from relpomdp.oopomdp.framework import ObjectObservation,\
    OOObservationModel, OOTransitionModel, CompositeRewardModel, OOBelief,\
    Objstate
import numpy as np

MOTION_ACTIONS = {MoveN, MoveS, MoveE, MoveW}


class NKAgent:
    """This agent is not a pomdp_py.Agent but it can
    be isntantiated into one.

    It begins with the knowledge of the given (partial) map,
    and the knowledge that it can move in the world (MoveEffect).

    It does not have purpose to begin with. But, one can
    given the agent a sensor (add_sensor), a new action, with
    a condition effect pair (add_action), a new objective (that is,
    a new reward function), and a new belief distribution about some object.
    """
    def __init__(self, robot_id, init_robot_pose,
                 grid_map=None, all_motion_actions=MOTION_ACTIONS):
        """
        The robot does not know the map. But we are not doing SLAM;
        the robot basically expands a partial map with perfect locations.
        """
        # Initially, the robot's map is empty
        if grid_map is None:
            grid_map = PartialGridMap(set({init_robot_pose[:2]}), {})
        elif isinstance(grid_map, GridMap):
            grid_map = PartialGridMap(set((x,y)
                                          for x in range(grid_map.width)
                                          for y in range(grid_map.length)), grid_map.walls)
        self.robot_id = robot_id
        self.motion_actions = all_motion_actions
        self.grid_map = grid_map
        self.legal_motions = self.grid_map.compute_legal_motions(self.motion_actions)

        # It knows how to move and its effect
        self.move_condition = CanMove(robot_id, self.legal_motions)
        self.move_effect = MoveEffect(robot_id)

        # It begins with no sensor. This dict maps from sensor name to (sensor, noise_params)
        self._sensors = {}
        self._sensor_caches = {}  # maps from sensor name to SensorCache

        # It begins with no other actions and effects besides move
        # This is a set of (actions, (cond, eff)) that stores
        # the actions and their corresponding cond/effect;
        # it is possible to have an action having two different effects.
        self._t = [(self.motion_actions, (self.move_condition, self.move_effect))]

        # It begins with an empty reward function; This list is used
        # to construct a CompositeRewardModel when instantiating an Agent
        self._reward_models = []

        # It begins with no belief about other objects other than its own pose
        # We require that these beliefs are up-to-date when instantiating
        # an agent; i.e. the instantiated agent's belief comes from these beliefs.
        init_robot_state = Objstate("Robot", pose=init_robot_pose)
        self._object_beliefs = {self.robot_id: Histogram({init_robot_state: 1.0})}

    @property
    def object_beliefs(self):
        return self._object_beliefs

    @property
    def sensors(self):
        return self._sensors

    @property
    def sensor_caches(self):
        return self._sensor_caches

    def all_actions(self):
        """Returns the set of unique actions at this point"""
        all_actions = set()
        for actions, cond_eff in self._t:
           all_actions.update(actions)
        return all_actions

    def add_sensor(self, sensor, noise_params):
        if sensor.name in self._sensors:
            raise ValueError("Sensor %s is already added." % sensor.name)
        self._sensors[sensor.name] = (sensor, noise_params)
        self._sensor_caches[sensor.name] = SensorCache(sensor.name)

    def add_actions(self, actions, condition, effect):
        """Add (multiple) actions, and their condition / effect.
        It's totally cool to add just a single action."""
        self._t.append((actions, (condition, effect)))

    def add_reward_model(self, reward_model):
        self._reward_models.append(reward_model)

    def set_belief(self, objid, belief):
        self._object_beliefs[objid] = belief

    def object_belief(self, objid):
        return self._object_beliefs[objid]

    def sensors_for(self, objclass):
        result = set()
        for sensor_name in self._sensors:
            noise_params = self._sensors[sensor_name][1]
            if objclass in noise_params:
                result.add(sensor_name)
        return result

    def check_integrity(self):
        """Check if this agent is up-to-date / behaves correctly
        This means: At any time, you can expect
        - The self.legal_motions is exactly the legal motions
          you would obtain if you compute this based on the self.grid_map."""
        # Check if the legal motions match.
        assert self.legal_motions == self.grid_map.compute_legal_motions(self.motion_actions)
        assert self.move_condition.legal_motions == self.legal_motions
        # There should be at most one reward model for each target id
        targets = set()
        for reward_model in self._reward_models:
            if hasattr(reward_model, "target_id"):
                assert reward_model.target_id not in targets, "duplicated reward model for %d" % reward_model.target_id
                targets.add(reward_model.target_id)

    def remove_reward_model(self, target_id):
        """TODO: Right now the reward model is removed
        based on target_id but this is not applicable to more types of reward models."""
        self._reward_models = [reward_model
                               for reward_model in self._reward_models
                               if reward_model.target_id != target_id]

    def build_observation_model(self, sensors_in_use=None, grid_map=None, caches=None):
        """Build an observation model for a given subset of sensors"""
        if grid_map is None:
            grid_map = self.grid_map

        if sensors_in_use is None:
            sensors_in_use = self._sensors

        cond_effects = []
        for name in sensors_in_use:
            sensor, noise_params = self._sensors[name]
            sensor_cache = None
            if self._sensor_caches[sensor.name].map_serving == grid_map.name:
                sensor_cache = self._sensor_caches[sensor.name]
            elif caches is not None:
                if caches[name].map_serving == grid_map.name:
                    sensor_cache = caches[name]

            observe_cond = CanObserve()
            observe_eff = ObserveEffect(
                self.robot_id, sensor,
                grid_map, noise_params,
                sensor_cache=sensor_cache)
            cond_effects.append((observe_cond, observe_eff))

        observation_model = OOObservationModel(cond_effects)
        return observation_model

    def instantiate(self, policy_model,
                    sensors_in_use=None,
                    objects_tracking=None):
        """
        The user who instantiates this NKAgent is responsible
        for providing a policy model. Because, the user should
        maintain what kind of preferred rollout policy should
        be used for this agent because that depends on the task
        the user is implementing, which the NKAgent is not aware of.

        The instantiation returns a pomdp_py.Agent with the T/O/R/pi models,
        and an initial belief.

        Args:
            policy_model (PolicyModel): A policy model to give to the agent.
                Note that, it should contains a `.actions` property, which
                defines the valid actions considered by the isntantiated agent;
                The agent's transition model will be built only for the valid
                set of actions.
            sensors_in_use (set): a set of sensor names used. Default is None,
                where all sensors are used
            objects_tracking  (set): A set of object ids whose beliefs will be
                passed on to the instantiated Agent.
        """
        if objects_tracking is None:
            objects_tracking = self._object_beliefs.keys()

        init_belief = OOBelief({objid:self._object_beliefs[objid]
                                for objid in objects_tracking})

        # Transition model
        t_condeff = [tup[1] for tup in self._t
                     if len(tup[0] & policy_model.actions) > 0]
        transition_model = OOTransitionModel(t_condeff)

        # Observation model
        observation_model = self.build_observation_model(sensors_in_use)

        # Reward model
        reward_model = CompositeRewardModel(self._reward_models)

        # Build Agent
        agent = pomdp_py.Agent(init_belief,
                               policy_model,
                               transition_model,
                               observation_model,
                               reward_model)
        agent.grid_map = self.grid_map
        return agent
