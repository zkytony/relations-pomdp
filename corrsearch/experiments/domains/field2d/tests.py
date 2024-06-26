import numpy as np
import matplotlib.pyplot as plt
import unittest
from corrsearch.experiments.domains.field2d.detector import *
from corrsearch.experiments.domains.field2d.problem import Field2D
from corrsearch.experiments.domains.field2d.parser import *
from corrsearch.models import *
from corrsearch.utils import *
from scipy import stats
import cv2
import math

class TestRangeDetector(unittest.TestCase):

    def setUp(self):
        self.robot_id = 0
        disk_sensor = DiskSensor(radius=1)
        self.label_detector = RangeDetector(100, self.robot_id,
                                            detection_type="label",
                                            true_positive={10:0.8, 14:0.8},
                                            false_positive={10:0.1, 14:0.1},
                                            sensors={10:disk_sensor, 14:disk_sensor})
        self.loc_detector = RangeDetector(200, self.robot_id,
                                          detection_type="loc",
                                          true_positive={10:0.7, 14:0.7},
                                          false_positive={10:0.1, 14:0.1},
                                          sigma={10:0.9, 14:0.1},
                                          sensors={10:disk_sensor, 14:disk_sensor})

    def test_label_detector(self):
        objstate_1 = LocObjState(10, "box", {"loc": (5,4)})
        objstate_2 = LocObjState(14, "box", {"loc": (8,4)})
        robot_state = RobotState(self.robot_id, {"loc": (6,4)})
        action = UseDetector(self.label_detector.id)
        # True positive detection should equal to true positive rate
        tps = []
        fps = []
        total_count = 0
        for i in range(5000):
            obz1 = self.label_detector.isample(objstate_1, robot_state, action)
            obz2 = self.label_detector.isample(objstate_2, robot_state, action)
            if isinstance(obz1, LabelObz):
                tps.append(obz1)
                self.assertEqual(self.label_detector.iprob(obz1, objstate_1, robot_state, action),
                                 self.label_detector.params["true_positive"][objstate_1.id])
            if isinstance(obz2, LabelObz):
                fps.append(obz2)
                self.assertEqual(self.label_detector.iprob(obz2, objstate_2, robot_state, action),
                                 self.label_detector.params["false_positive"][objstate_2.id])
            total_count += 1

        self.assertAlmostEqual(len(tps)/total_count,
                               self.label_detector.params["true_positive"][objstate_1.id],
                               1)
        self.assertAlmostEqual(len(fps)/total_count,
                               self.label_detector.params["false_positive"][objstate_2.id],
                               1)

    def test_loc_detector(self):
        objstate_1 = LocObjState(10, "box", {"loc": (5,4)})
        objstate_2 = LocObjState(14, "box", {"loc": (8,4)})
        robot_state = RobotState(self.robot_id, {"loc": (6,4)})
        action = UseDetector(self.loc_detector.id)
        # True positive detection should equal to true positive rate
        tps = []
        fps = []
        total_count = 0
        for i in range(5000):
            obz1 = self.loc_detector.isample(objstate_1, robot_state, action)
            obz2 = self.loc_detector.isample(objstate_2, robot_state, action)
            if isinstance(obz1, LocObz):
                tps.append(obz1["loc"])

            if isinstance(obz2, LocObz):
                fps.append(obz2["loc"])
                size = self.loc_detector.sensor_region_size(objstate_2.id, robot_state)
                expected_pr = self.loc_detector.params["false_positive"][objstate_2.id] * (1/size)
                self.assertEqual(self.loc_detector.iprob(obz2, objstate_2,
                                                          robot_state, action),
                                  expected_pr)
            total_count += 1

        # True positive should follow a gaussian
        self.assertEqual(tuple(stats.mode(tps).mode.tolist()[0]), objstate_1["loc"])
        self.assertAlmostEqual(len(tps)/total_count,
                               self.loc_detector.params["true_positive"][objstate_1.id],
                               1)
        self.assertAlmostEqual(len(fps)/total_count,
                               self.loc_detector.params["false_positive"][objstate_1.id],
                               1)

    def test_disk_sensor_geometry(self):
        w = 10
        l = 10
        locations = [(x,y) for x in range(w) for y in range(l)]
        robot_pose = (5, 5, to_rad(0))
        accepted = set()
        sensor = DiskSensor(radius=1)
        for point in locations:
            if sensor.in_range(point, robot_pose):
                accepted.add(point)
        counts = {}
        for i in range(1000):
            point = sensor.uniform_sample_sensor_region(robot_pose)
            self.assertTrue(point in accepted)
            counts[point] = counts.get(point, 0) + 1
        self.assertEqual(len(counts), sensor.sensor_region_size)

    def test_laser_sensor_geometry(self):
        w = 10
        l = 10
        locations = [(x,y) for x in range(w) for y in range(l)]
        robot_pose = (5, 5, to_rad(0))
        accepted = set()
        sensor = FanSensor(fov=50, min_range=0, max_range=4)
        for point in locations:
            if sensor.in_range(point, robot_pose):
                accepted.add(point)
        counts = {}
        for i in range(1000):
            point = sensor.uniform_sample_sensor_region(robot_pose)
            self.assertTrue(point in accepted)
            counts[point] = counts.get(point, 0) + 1
        self.assertEqual(len(counts), sensor.sensor_region_size)


class TestField2DProblem(unittest.TestCase):
    def setUp(self):
        self.problem = problem_from_file("./configs/simple_config.yaml")

    def test_deterministic_random_world(self):
        """Tests that the `seed` makes the random generation
        of the world deterministic"""
        instance_config = dict(
            init_locs="random",
            init_robot_setting=((0, 0, 0), 100),
            init_belief="uniform",
            seed=100
        )
        env, agent = self.problem.instantiate(**instance_config)
        for i in range(10):
            env2, agent2 = self.problem.instantiate(**instance_config)
            self.assertEqual(env.state, env2.state)

    def test_idential_enumeration(self):
        instance_config = dict(
            init_locs="random",
            init_robot_setting=((0, 0, 0), 100),
            init_belief="uniform",
            seed=100
        )
        env, agent = self.problem.instantiate(**instance_config)
        self.assertEqual(agent.all_states, agent.all_states)
        self.assertEqual(agent.all_actions, agent.all_actions)
        self.assertEqual(agent.all_observations, agent.all_observations)
        print("|S| = %d" % len(agent.all_states))
        print("|A| = %d" % len(agent.all_actions))
        print("|Z| = %d" % len(agent.all_observations))



# Test detector geometry by plotting
def plot_disk_sensor_geometry():
    w = 10
    l = 10
    locations = [(x,y) for x in range(w) for y in range(l)]
    robot_pose = (5, 5, to_rad(0))
    accepted_x = []
    accepted_y = []

    sensor = DiskSensor(radius=1)
    for point in locations:
        if sensor.in_range(point, robot_pose):
            accepted_x.append(point[0])
            accepted_y.append(point[1])
    ax = plt.gca()
    plt.scatter(accepted_x, accepted_y)
    plot_pose(ax, robot_pose[0:2], robot_pose[2])
    ax.set_xlim(0, w)
    ax.set_ylim(0, l)
    plt.show(block=False)
    plt.pause(1.5)
    plt.close()

def plot_laser_sensor_geometry():
    w = 10
    l = 10
    locations = [(x,y) for x in range(w) for y in range(l)]
    robot_pose = (5, 5, to_rad(18))
    accepted_x = []
    accepted_y = []
    sensor = FanSensor(fov=75, min_range=0, max_range=4)
    for point in locations:
        if sensor.in_range(point, robot_pose):
            accepted_x.append(point[0])
            accepted_y.append(point[1])
    ax = plt.gca()
    plt.scatter(accepted_x, accepted_y, zorder=2)
    plot_pose(ax, robot_pose[0:2], robot_pose[2])
    ax.set_xlim(0, w)
    ax.set_ylim(0, l)
    ax.set_aspect("equal")

    samples_x = []
    samples_y = []
    for i in range(1000):
        point = sensor.uniform_sample_sensor_region(robot_pose)
        samples_x.append(point[0])
        samples_y.append(point[1])
    plt.scatter(samples_x, samples_y, zorder=1, s=50)
    plt.show(block=False)
    plt.pause(1.5)
    plt.close()

def test_field2d_visualize():
    problem = problem_from_file("./configs/simple_config.yaml")
    viz = problem.visualizer(bg="./imgs/whitefloor.jpeg", res=30)

    objstate = ObjectState(1, "blue-cube", {"loc": (0,1)})
    objstate2 = ObjectState(2, "red-cube", {"loc": (1,0)})
    objstate3 = ObjectState(0, "robot", {"loc": (0,0),
                                         "pose": (0,0,0)})
    state = JointState({1:objstate, 2:objstate2, 0:objstate3})

    img = viz.visualize(state)
    # These operations are necessary for imshow to match
    # pygame's display
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)  # rotate 90deg clockwise
    img = cv2.flip(img, 0)  # flip horizontally
    cv2.imshow("TT", img)  # imshow uses BGR, while pygame uses RGB
    cv2.waitKey(3000)

if __name__ == "__main__":
    unittest.main(exit=False)
    # plot_disk_sensor_geometry()
    # plot_laser_sensor_geometry()
    # test_field2d_visualize()
