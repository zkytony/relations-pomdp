# nk_agent: Agent without prior knowledge of the map

import pomdp_py
from pomdp_py import Histogram
from relpomdp.home2d.domain.maps.grid_map import GridMap
from relpomdp.home2d.domain.action import *
from relpomdp.home2d.agent.observation_model import CanObserve, ObserveEffect
from relpomdp.home2d.agent.transition_model import CanPickup, PickupEffect, Pickup
from relpomdp.home2d.agent.reward_model import PickupRewardModel
from relpomdp.home2d.tasks.common.sensor import Laser2DSensor
from relpomdp.home2d.domain.condition_effect import CanMove, MoveEffect
from relpomdp.oopomdp.framework import ObjectObservation,\
    OOObservationModel, OOTransitionModel, CompositeRewardModel, OOBelief,\
    Objstate
import numpy as np

MOTION_ACTIONS = {MoveN, MoveS, MoveE, MoveW}

class PartialGridMap(GridMap):

    def __init__(self, free_locations, walls):
        """
        free_locations (set): a set of (x,y) locations that are free
        walls (dict): map from objid to WallState
        """
        self.free_locations = free_locations
        self.walls = walls
        self._cell_to_walls = self._compute_walls_per_cell()

        width, length = self._compute_dims(free_locations)
        super().__init__(width, length, walls, {})

    def _compute_dims(self, free_locs):
        width = (max(point[0] for point in free_locs) - min(point[0] for point in free_locs)) + 1
        length = (max(point[1] for point in free_locs) - min(point[1] for point in free_locs)) + 1
        return width, length

    def _compute_walls_per_cell(self):
        """
        Returns a map that maps from each location to a set of walls (wall ids)
        that surrounds this cell
        """
        cell_to_walls = {}
        for wall_id in self.walls:
            cell1, cell2 = self.walls[wall_id].cells_touching()
            if cell1 not in cell_to_walls:
                cell_to_walls[cell1] = set()
            if cell2 not in cell_to_walls:
                cell_to_walls[cell2] = set()
            cell_to_walls[cell1].add(wall_id)
            cell_to_walls[cell2].add(wall_id)
        return cell_to_walls


    def update(self, free_locs, walls):
        self.free_locations |= free_locs
        self.walls.update(walls)
        self.width, self.length = self._compute_dims(self.free_locations)
        self._cell_to_walls = self._compute_walls_per_cell()

    def frontier(self):
        """Returns a set of locations that is an immediate
        expansion of locations at the edge of the current map"""
        frontier = set()
        for x, y in self.free_locations:
            # Check all four directions of this grid cell and
            # see if there is one side that extends into the unknown
            connecting_cells = {(x+1, y), (x-1,y), (x,y+1), (x,y-1)}
            if (x,y) in self._cell_to_walls:
                surrounding_walls = self._cell_to_walls[(x,y)]
                for wall_id in surrounding_walls:
                    wall = self.walls[wall_id]
                    blocked_loc = set(wall.cells_touching()) - set({(x,y)})
                    connecting_cells -= blocked_loc

            for cell in connecting_cells:
                if cell not in self.free_locations\
                   and (cell[0] >= 0 and cell[1] >= 0):
                    # This is a frontier, because it is not blocked by
                    # any wall and is not in a free location, and it
                    # does not have negative coordinates
                    frontier.add(cell)
        return frontier

    def compute_legal_motions(self, all_motion_actions):
        """This is done by creating a map from
        current free locations and frontier to a set of
        motions that can be executed there."""
        legal_actions = {}
        all_locations = self.free_locations | self.frontier()
        for x, y in all_locations:
            legal_actions[(x,y)] = self.legal_motions_at(x, y, all_motion_actions,
                                                         permitted_locations=all_locations)
        return legal_actions


class FakeSLAM:
    def __init__(self, range_sensor):
        """
        range_sensor provides the field of view ( e.g. Laser2DSensor)
        """
        self.range_sensor = range_sensor

    def update(self, partial_map, prev_robot_pose, robot_pose, env):
        """
        Projects the range sensor from the robot pose, and get readings
        based on environment's full map. Then update the partial map
        based on such readings.

        We want to simulate the process of robot moving from a previous
        robot pose to a current one so it will observe the walls in between.
        For us it is straightforward; First update the angle, then update the x,y pose.
        """
        full_grid_map = env.grid_map
        free_locs = set()
        walls = {}
        # We want to simulate the process of the robot
        interm_pose = prev_robot_pose[:2] + (robot_pose[2],)
        for x in np.arange(-1, full_grid_map.width+1, 1):
            for y in np.arange(-1, full_grid_map.length+1, 1):

                res1, wall1 = self.range_sensor.within_range(
                    interm_pose, (x,y), grid_map=full_grid_map,
                    return_intersecting_wall=True)
                # res2, wall2 = False, None
                res2, wall2 = self.range_sensor.within_range(
                    robot_pose, (x,y), grid_map=full_grid_map,
                    return_intersecting_wall=True)

                if res1 or res2:
                    free_locs.add((x,y))
                else:
                    if wall1 is not None:
                        # The point is blocked by some wall that is in the FOV
                        # TODO: REFACTOR: Getting wall id should not be necessary
                        wall_id, wall_state = wall1
                        walls[wall_id] = wall_state
                    if wall2 is not None:
                        # The point is blocked by some wall that is in the FOV
                        # TODO: REFACTOR: Getting wall id should not be necessary
                        wall_id, wall_state = wall2
                        walls[wall_id] = wall_state
        partial_map.update(free_locs, walls)


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

        # It begins with no sensor. This dict maps from sensor name to (sensor, (cond, eff))
        self._sensors = {}

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

    def all_actions(self):
        """Returns the set of unique actions at this point"""
        all_actions = set()
        for actions, cond_eff in self._t:
           all_actions.update(actions)
        return all_actions

    def add_sensor(self, sensor, noise_params, gamma=1.0):
        if sensor.name in self._sensors:
            raise ValueError("Sensor %s is already added." % sensor.name)
        observe_cond = CanObserve()
        observe_eff = ObserveEffect(self.robot_id, sensor, self.grid_map, noise_params,
                                    gamma=gamma)
        self._sensors[sensor.name] =\
            (sensor, (observe_cond, observe_eff))

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
            eff = self._sensors[sensor_name][1][1]
            if objclass in eff.noise_params:
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
                assert reward_model.target_id not in targets, "duplicated reward model for %d" % target_id
                targets.add(reward_model.target_id)

    def remove_reward_model(self, target_id):
        """TODO: Right now the reward model is removed
        based on target_id but this is not applicable to more types of reward models."""
        self._reward_models = [reward_model
                               for reward_model in self._reward_models
                               if reward_model.target_id != target_id]

    def build_observation_model(self, sensors_in_use=None):
        if sensors_in_use is None:
            sensors_in_use = self._sensors
        o_condeff = [self._sensors[name][1]
                     for name in sensors_in_use]
        observation_model = OOObservationModel(o_condeff)
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


    # def add_target(self, target_id, target_class, init_belief):
    #     """
    #     Adds target to search for, with an initial belief given.
    #     """
    #     pickup_condeff = (CanPickup(self.robot_id, target_id),
    #                       PickupEffect())
    #     self._transition_model.cond_effects.append(pickup_condeff)
    #     self._reward_model.add_model(PickupRewardModel(self.robot_id, target_id))
    #     self.add_belief(target_id, target_class, init_belief)

    # def add_belief(self, objid, objclass, init_belief):
    #     """
    #     Expands belief to include one more object
    #     """
    #     self._init_belief.object_beliefs[objid] = init_belief



    # def update(self, robot_pose=None, prev_robot_pose=None, action=None):
    #     """After the map is updated, the policy model and the observation models
    #     should be updated; But the observation model should be updated automatically
    #     because we passed in the reference to self.grid_map when constructing it."""
    #     legal_motions = self.grid_map.compute_legal_motions(self.motion_actions)
    #     # action_prior = ExplorationActionPrior(self.robot_id, self.grid_map,
    #     #                                       legal_motions,
    #     #                                       10, 100)
    #     # self._policy_model = PreferredPolicyModel(action_prior,
    #     #                                           other_actions={Pickup()})
    #     self._transition_model.cond_effects.pop(0)  # pop the MoveEffect
    #     move_condeff = (CanMove(self.robot_id, legal_motions), MoveEffect(self.robot_id))
    #     self._transition_model.cond_effects.insert(0, move_condeff)
    #     memory = {} if self._policy_model is None else self._policy_model.memory
    #     self._policy_model = PolicyModel(self.robot_id,
    #                                      motions=self.motion_actions,
    #                                      other_actions={Pickup()},
    #                                      grid_map=self.grid_map,
    #                                      memory=memory)
    #     if robot_pose is not None:
    #         self._policy_model.update(robot_pose, prev_robot_pose, action)  # This records invalid acitons
