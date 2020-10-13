# nk_agent: Agent without prior knowledge of the map

import pomdp_py
from relpomdp.home2d.domain.maps.grid_map import GridMap
from relpomdp.home2d.domain.action import *
from relpomdp.oopomdp.framework import ObjectObservation
from relpomdp.home2d.tasks.common.sensor import Laser2DSensor

class PartialGridMap:

    def __init__(self, free_locations, walls):
        """
        free_locations (set): a set of (x,y) locations that are free
        walls (dict): map from objid to WallState
        """
        self.free_locations = free_locations
        self.walls = walls

    def update(self, free_locs, walls):
        self.free_locations |= free_locs
        self.walls.update(walls)

    def legal_motions(self, x, y, all_motion_actions={MoveN, MoveS, MoveE, MoveW}):
        motion_actions = set(all_motion_actions)
        so = Objstate("PoseObject", pose=(x,y))
        for wall_id in self.walls:
            wall = self.walls[wall_id]
            if touch_N(so, wall):
                motion_actions.remove(MoveN)
            elif touch_S(so, wall):
                motion_actions.remove(MoveS)
            elif touch_E(so, wall):
                motion_actions.remove(MoveE)
            elif touch_W(so, wall):
                motion_actions.remove(MoveW)
        return motion_actions


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


class NKAgent(pomdp_py.Agent):
    def __init__(self, init_robot_pose):
        """
        The robot does not know the map. But we are not doing SLAM;
        the robot basically expands a partial map with perfect locations.
        """
        # Initially, the robot's map is empty
        self.grid_map = PartialGridMap(set({init_robot_pose[:2]}), {})
