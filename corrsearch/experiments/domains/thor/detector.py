from corrsearch.experiments.domains.field2d.detector import *
import numpy as np
import math


class FanSensorThor(FanSensor):

    def __init__(self, name="laser2d_sensor", **params):
        super().__init__(name=name, **params)

        if "grid_map" not in params:
            raise ValueError("Thor Fan Sensor needs grid map")
        self.grid_map = params["grid_map"]
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
            # The point is within the fan shape. But, it may be
            # out of bound, or occluded by an obstacle. First, obtain
            # unit vector from robot to point
            rp = np.array(robot_pose[:2])
            rx, ry = rp
            px, py = point
            vec = np.array([px - rx, py - ry]).astype(float)
            vec /= np.linalg.norm(vec)

            # Check points along the line from robot pose to the point
            result = True
            nsteps = 20
            dist = euclidean_dist(point, (rx,ry))
            step_size = dist / nsteps
            t = 0
            while t < nsteps:
                line_point = tuple(np.round(rp + (t*step_size*vec)).astype(int))
                if line_point in self.grid_map.obstacles:
                    result = False
                    break
                t += 1

        self._cache[(point, robot_pose)] = result
        return result

    def shoot_beam(self, robot_pose, point):
        """Shoots a beam from robot_pose at point. Returns the distance and bearing
        of the beame (i.e. the length and orientation of the beame)"""
        rx, ry, rth = robot_pose
        dist = euclidean_dist(point, (rx,ry))
        bearing = (math.atan2(point[0] - rx, point[1] - ry) - rth) % (2*math.pi)  # bearing (i.e. orientation)
        return (dist, bearing)
