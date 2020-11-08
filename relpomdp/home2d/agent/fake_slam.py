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

        We want to simulate the process of robot moving from a previous
        robot pose to a current one so it will observe the walls in between.
        For us it is straightforward; First update the angle, then update the x,y pose.
        """
        full_grid_map = env.grid_map
        if not self.sensor_cache.serving(full_grid_map):
            raise ValueError("FakeSLAM's cache is being used to serve for map %s" % cache.map_serving)

        free_locs = set()
        walls = {}
        loc_to_room = {}
        # We want to simulate the process of the robot
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
        partial_map.update(free_locs, walls, loc_to_room=loc_to_room)
