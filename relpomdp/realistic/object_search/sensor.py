# """Sensor model (for example, laser scanner)"""
from relpomdp.realistic.utils.util import euclidean_dist
import math
import random

# A simpler fan-shaped sensor
class FanSensor:
    """2D fanshape sensor"""
    def __init__(self, fov=90, min_range=0.01, max_range=0.50):
        self.fov = math.radians(fov)
        self.min_range = min_range
        self.max_range = max_range

    def within_range(self, robot_pose, point):
        (x, z), th = robot_pose
        dist = euclidean_dist((x,z), point)
        if self.min_range <= dist <= self.max_range:
            bearing = math.atan2(point[1] - x, point[0] - z) - math.radians(th)
            if (-self.fov/2 <= bearing <= 0) or (0 <= bearing <= self.fov/2):
                return True
            else:
                return False
        return False

def test():
    # Case 1
    fov = 90
    min_range = 0.0
    max_range = 1.0
    robot_pose = ((0.0, 0.0), 0.0)
    sensor = FanSensor(fov=fov, min_range=min_range, max_range=max_range)

    assert sensor.within_range(robot_pose, (1,0)) == True
    assert sensor.within_range(robot_pose, (0,1)) == False
    assert sensor.within_range(robot_pose, (0.5, 0)) == True
    assert sensor.within_range(robot_pose, (0, 0.5)) == False
    assert sensor.within_range(robot_pose, (0,-0.5)) == False
    assert sensor.within_range(robot_pose, (0.6, -0.3)) == True

if __name__ == "__main__":
    test()
