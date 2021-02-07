import numpy as np
import matplotlib.pyplot as plt
import unittest
from corrsearch.experiments.domains.field2d.detector import *
from corrsearch.models import *
from scipy import stats


class ToyRangeDetector(RangeDetector):
    def sensor_region(self, objid, robot_state):
        return set([tuple(np.arange(robot_state["loc"][0]-1, robot_state["loc"][0]+1)),
                    tuple(np.arange(robot_state["loc"][1]-1, robot_state["loc"][1]+1))])

    def in_range(self, objstate, robot_state):
        x, y = objstate["loc"]
        rx, ry = robot_state["loc"]
        return abs(rx - x) <= 1 and abs(ry - y) <= 1

class TestRangeDetector(unittest.TestCase):

    def setUp(self):
        self.robot_id = 0
        self.label_detector = ToyRangeDetector(100, self.robot_id,
                                               detection_type="label",
                                               true_positive=0.8,
                                               false_positive=0.1)
        self.loc_detector = ToyRangeDetector(200, self.robot_id,
                                             detection_type="loc",
                                             true_positive=0.7,
                                             false_positive=0.1,
                                             sigma=0.9)

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
                self.assertEquals(self.label_detector.iprob(obz1, objstate_1, robot_state, action),
                                  self.label_detector.params["true_positive"])
            if isinstance(obz2, LabelObz):
                fps.append(obz2)
                self.assertEquals(self.label_detector.iprob(obz2, objstate_2, robot_state, action),
                                  self.label_detector.params["false_positive"])
            total_count += 1

        self.assertAlmostEqual(len(tps)/total_count,
                               self.label_detector.params["true_positive"],
                               1)
        self.assertAlmostEqual(len(fps)/total_count,
                               self.label_detector.params["false_positive"],
                               2)

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
                region = self.loc_detector.sensor_region(objstate_2.id, robot_state)
                expected_pr = self.loc_detector.params["false_positive"] * (1/len(region))
                self.assertEquals(self.loc_detector.iprob(obz2, objstate_2,
                                                          robot_state, action),
                                  expected_pr)
            total_count += 1

        # True positive should follow a gaussian
        self.assertEqual(tuple(stats.mode(tps).mode.tolist()[0]), objstate_1["loc"])
        self.assertAlmostEqual(len(tps)/total_count,
                               self.loc_detector.params["true_positive"],
                               1)
        self.assertAlmostEqual(len(fps)/total_count,
                               self.loc_detector.params["false_positive"],
                               1)



if __name__ == "__main__":
    unittest.main()
