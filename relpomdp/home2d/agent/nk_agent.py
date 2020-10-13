# nk_agent: Agent without prior knowledge of the map

import pomdp_py
from relpomdp.home2d.domain.maps.grid_map import GridMap
from relpomdp.home2d.domain.action import *
from relpomdp.home2d.agent.observation_model import CanObserve, ObserveEffect
from relpomdp.home2d.agent.transition_model import CanPickup, PickupEffect, Pickup
from relpomdp.home2d.agent.reward_model import PickupRewardModel
from relpomdp.home2d.agent.policy_model import ExplorationActionPrior, PreferredPolicyModel, PolicyModel
from relpomdp.home2d.tasks.common.sensor import Laser2DSensor
from relpomdp.home2d.domain.condition_effect import CanMove, MoveEffect
from relpomdp.oopomdp.framework import ObjectObservation,\
    OOObservationModel, OOTransitionModel, CompositeRewardModel, OOBelief,\
    Objstate

MOTION_ACTIONS = {MoveN, MoveS, MoveE, MoveW}

class PartialGridMap(GridMap):

    def __init__(self, free_locations, walls):
        """
        free_locations (set): a set of (x,y) locations that are free
        walls (dict): map from objid to WallState
        """
        self.free_locations = free_locations
        self.walls = walls

        width, length = self._compute_dims(free_locations)
        super().__init__(width, length, walls, {})

    def _compute_dims(self, free_locs):
        width = (max(point[0] for point in free_locs) - min(point[0] for point in free_locs)) + 1
        length = (max(point[1] for point in free_locs) - min(point[1] for point in free_locs)) + 1
        return width, length

    def update(self, free_locs, walls):
        self.free_locations |= free_locs
        self.walls.update(walls)
        self.width, self.length = self._compute_dims(self.free_locations)


class FakeSLAM:
    def __init__(self, range_sensor):
        """
        range_sensor provides the field of view ( e.g. Laser2DSensor)
        """
        self.range_sensor = range_sensor

    def update(self, partial_map, robot_pose, env):
        """
        Projects the range sensor from the robot pose, and get readings
        based on environment's full map. Then update the partial map
        based on such readings.
        """
        full_grid_map = env.grid_map
        free_locs = set()
        walls = {}
        for x in range(full_grid_map.width):
            for y in range(full_grid_map.length):
                res, wall = self.range_sensor.within_range(
                    robot_pose, (x,y), grid_map=full_grid_map,
                    return_intersecting_wall=True)
                if res:
                    free_locs.add((x,y))
                else:
                    if wall is not None:
                        # The point is blocked by some wall that is in the FOV
                        # TODO: REFACTOR: Getting wall id should not be necessary
                        wall_id, wall_state = wall
                        walls[wall_id] = wall_state
        partial_map.update(free_locs, walls)


class NKAgent:
    """This agent is not a pomdp_py.Agent but it can
    be isntantiated into one."""
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
        self.grid_map = grid_map
        self.robot_id = robot_id
        self.motion_actions = all_motion_actions

        move_condeff = (CanMove(robot_id, None), MoveEffect(robot_id))
        self._transition_model = OOTransitionModel([move_condeff])
        self._observation_model = None
        self._reward_model = CompositeRewardModel([])
        self._policy_model = None

        init_robot_state = Objstate("Robot", pose=init_robot_pose)
        self._init_belief = OOBelief({self.robot_id: pomdp_py.Histogram({init_robot_state:1.0})})

    def instantiate(self):
        agent = pomdp_py.Agent(self._init_belief,
                               self._policy_model,
                               self._transition_model,
                               self._observation_model,
                               self._reward_model)
        agent.grid_map = self.grid_map
        return agent


    def add_sensor(self, sensor, noise_params):
        condeff = (CanObserve(),
                   ObserveEffect(self.robot_id, sensor, self.grid_map, noise_params))
        if self._observation_model is None:
            self._observation_model = OOObservationModel([condeff])
        else:
            self._observation_model.cond_effects.append(condeff)

    def add_target(self, target_id, target_class, init_belief):
        """
        Adds target to search for, with an initial belief given.
        """
        pickup_condeff = (CanPickup(self.robot_id, target_id),
                          PickupEffect())
        self._transition_model.cond_effects.append(pickup_condeff)
        self._reward_model.add_model(PickupRewardModel(self.robot_id, target_id))
        self.add_belief(target_id, target_class, init_belief)

    def add_belief(self, objid, objclass, init_belief):
        """
        Expands belief to include one more object
        """
        self._init_belief.object_beliefs[objid] = init_belief

    def update(self):
        """After the map is updated, the policy model and the observation models
        should be updated; But the observation model should be updated automatically
        because we passed in the reference to self.grid_map when constructing it."""
        legal_motions = self.grid_map.compute_legal_motions(self.motion_actions)
        # action_prior = ExplorationActionPrior(self.robot_id, self.grid_map,
        #                                       legal_motions,
        #                                       10, 10)
        # self._policy_model = PreferredPolicyModel(action_prior,
        #                                           other_actions={Pickup()})
        self._policy_model = PolicyModel(self.robot_id,
                                         motions=self.motion_actions,
                                         other_actions={Pickup()},
                                         grid_map=self.grid_map)
