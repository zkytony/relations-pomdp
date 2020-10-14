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

    def instantiate(self, init_belief=None):
        if init_belief is None:
            init_belief = self._init_belief
        agent = pomdp_py.Agent(init_belief,
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
        #                                       10, 100)
        # self._policy_model = PreferredPolicyModel(action_prior,
        #                                           other_actions={Pickup()})
        self._transition_model.cond_effects.pop(0)  # pop the MoveEffect
        move_condeff = (CanMove(self.robot_id, legal_motions), MoveEffect(self.robot_id))
        self._transition_model.cond_effects.insert(0, move_condeff)
        self._policy_model = PolicyModel(self.robot_id,
                                         motions=self.motion_actions,
                                         other_actions={Pickup()},
                                         grid_map=self.grid_map)
