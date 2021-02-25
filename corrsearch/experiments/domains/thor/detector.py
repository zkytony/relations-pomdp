from corrsearch.experiments.domains.field2d.detector import *
import numpy as np
import math
import yaml
import pickle
import os


class FanSensorThor(FanSensor):

    def __init__(self, name="laser2d_sensor", grid_size=0.25, **params):
        """
        Will interpret min, max ranges in meters. Convert them
        to grids using grid size
        """
        fov = params.get("fov", 90)
        min_range = int(round(params.get("min_range", 0.0) / grid_size))
        max_range = int(round(params.get("max_range", 1.5) / grid_size))
        super().__init__(name=name, fov=fov, min_range=min_range, max_range=max_range)

        self.grid_map = params.get("grid_map", None)
        self._cache = {}

    def uniform_sample_sensor_region(self, robot_pose):
        """Returns a location in the field of view
        uniformly at random. Expecting robot pose to
        have x, y, th, where th is in radians."""
        assert len(robot_pose) == 3,\
            "Robot pose must have x, y, th"
        # Sample a location (r,th) for the default robot pose
        th = random.uniform(0, self.fov) - self.fov/2
        r = random.uniform(self.min_range, self.max_range+1)
        y, x = pol2cart(r, th)
        # transform to robot pose
        y, x = np.matmul(R2d(robot_pose[2]), np.array([y,x])) # rotation
        x += robot_pose[0]  # translation dx
        y += robot_pose[1]  # translation dy
        point = x, y#int(x), int(y)
        if not self.in_range(point, robot_pose):
            return self.uniform_sample_sensor_region(robot_pose)
        else:
            return point

    def in_range(self, point, robot_pose):
        if (point, robot_pose) in self._cache:
            return self._cache[(point, robot_pose)]

        if not super().in_range(point, robot_pose):
            result = False
        else:
            if self.grid_map is None:
                print("WARNING: grid_map not specified for this"\
                      "sensor (%s)\n Obstacles ignored." % self.name)
                return True

            # The point is within the fan shape. But, it may be
            # out of bound, or occluded by an obstacle. First, obtain
            # unit vector from robot to point
            result = not self.grid_map.blocked(robot_pose[:2], point)

        self._cache[(point, robot_pose)] = result
        return result

    def shoot_beam(self, robot_pose, point):
        """Shoots a beam from robot_pose at point. Returns the distance and bearing
        of the beame (i.e. the length and orientation of the beame)"""
        rx, ry, rth = robot_pose
        dist = euclidean_dist(point, (rx,ry))
        bearing = (math.atan2(point[0] - rx, point[1] - ry) - rth) % (2*math.pi)  # bearing (i.e. orientation)
        return (dist, bearing)




if __name__ == "__main__":
    # Little test for parsing detectors
    robot_id = 0
    detectors = parse_detector("FloorPlan_Train1_1", "./config/detectors_spec.yaml", robot_id)
    print(detectors)
