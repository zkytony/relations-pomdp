from relpomdp.home2d.agent.sensor import SensorCache
import numpy as np

class FakeSLAM:
    def __init__(self, range_sensor):
        """
        range_sensor provides the field of view ( e.g. Laser2DSensor)
        """
        self.range_sensor = range_sensor

        # This cache is used by the SLAM's sensor in the complete grid map.
        self.sensor_cache = SensorCache(self.range_sensor.name)

    def update(self,
               partial_map, prev_robot_pose,
               robot_pose, env):
        """
        Projects the range sensor from the robot pose, and get readings
        based on environment's full map. Then update the partial map
        based on such readings.

        If there is a wall immediately next to the robot, even though
        it falls outside of the FOV, the robot will sense it.
        """
        full_grid_map = env.grid_map
        if not self.sensor_cache.serving(full_grid_map):
            raise ValueError("FakeSLAM's cache is being used to serve for map %s" % cache.map_serving)

        free_locs = set()
        walls = {}
        loc_to_room = {}
        for x in np.arange(-1, full_grid_map.width+1, 1):
            for y in np.arange(-1, full_grid_map.length+1, 1):
                res, wall = self.range_sensor.within_range(
                    robot_pose, (x,y),
                    grid_map=full_grid_map, cache=self.sensor_cache,
                    return_intersecting_wall=True)

                if res:
                    free_locs.add((x,y))
                    loc_to_room[(x,y)] = full_grid_map.room_of((x,y))
                else:
                    if wall is not None:
                        # The point is blocked by some wall that is in the FOV
                        # TODO: REFACTOR: Getting wall id should not be necessary
                        wall_id, wall_state = wall
                        walls[wall_id] = wall_state
        # Get touching walls
        for wall_id in full_grid_map.walls:
            wall_state = full_grid_map.walls[wall_id]
            wx, wy = wall_state["pose"]
            rx, ry = robot_pose[:2]
            if wx == rx and wy == ry:
                touching = True
            else:
                if wall_state["direction"] == "V":
                    touching = wx + 1 == rx and wy == ry
                else:
                    touching = wx == rx and wy + 1== ry
            if touching:
                walls[wall_id] = wall_state
        partial_map.update(free_locs, walls, loc_to_room=loc_to_room)
