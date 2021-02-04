import unittest
import random
from corrsearch.models import *
from corrsearch.objects import *
from corrsearch.utils import *
from corrsearch.probability import *

# A toy detector model
class EvenObjDetector(DetectorModel):
    """This detector is only able to detect object with even id,
    and it detects them if robot is within a radius of them."""
    def __init__(self, detector_id, robot_id, radius=2):
        self.radius = 2
        super().__init__(detector_id, robot_id)

    def iprob(self, objobz, objstate, robot_state, action):
        """
        Returns the probability of Pr(zi | si, sr', a)
        """
        assert isinstance(action, UseDetector)
        assert action.detector_id == self.id
        if objobz.objid % 2 == 0:
            if euclidean_dist(objstate.loc, robot_state.loc) <= self.radius:
                return indicator(not isinstance(objobz, NullObz)\
                                 and objobz["loc"] == objstate.loc)
            else:
                return indicator(isinstance(objobz, NullObz))
        return 0.0

    def isample(self, objstate, robot_state, action):
        """
        Returns a sample zi according to Pr(zi | si, sr', a)
        """
        assert isinstance(action, UseDetector)
        assert action.detector_id == self.id
        if objstate.objid % 2 == 0:
            if euclidean_dist(objstate.loc, robot_state.loc) <= self.radius:
                return ObjectObz(objstate.id,
                                 objstate.objclass,
                                 {"loc":objstate.loc})
        return NullObz(objstate.id)


class TestDetector(unittest.TestCase):
    def setUp(self):
        self.robot_id = 0
        self.detector = EvenObjDetector(100, self.robot_id, radius=2)

    def test_sample_normal(self):
        objstate = LocObjState(10, "box", {"loc": (5,)})
        robot_state = RobotState(self.robot_id, {"loc": (7,)})
        action = UseDetector(self.detector.id)
        expected_objz = ObjectObz(objstate.id,
                                  objstate.objclass,
                                  {"loc":objstate.loc})
        self.assertEqual(self.detector.isample(objstate, robot_state, action),
                         expected_objz)
        self.assertEqual(self.detector.iprob(expected_objz, objstate, robot_state, action), 1.0)
        joint_state = JointState({objstate.id:objstate,
                                  robot_state.id:robot_state})
        expected_obz = JointObz({expected_objz.id: expected_objz})
        self.assertEqual(self.detector.sample(joint_state, action),
                         expected_obz)
        self.assertEqual(self.detector.probability(expected_obz, joint_state, action), 1.0)


    def test_sample_null(self):
        objstate = LocObjState(10, "box", {"loc": (5,)})
        robot_state = RobotState(self.robot_id, {"loc": (7,)})

        # If detection action doesn't match the detector then NullObz is returned also
        expected_obz = JointObz({objstate.id: NullObz(objstate.id)})

        action = UseDetector(self.detector.id+1)
        joint_state = JointState({objstate.id:objstate,
                                  robot_state.id:robot_state})
        self.assertEqual(self.detector.sample(joint_state, action),
                         expected_obz)
        self.assertEqual(self.detector.probability(expected_obz, joint_state, action),
                         1.0)

        # If object outside of region also Null, but
        objstate = LocObjState(10, "box", {"loc": (0,)})
        action = UseDetector(self.detector.id)
        joint_state = JointState({objstate.id:objstate,
                                  robot_state.id:robot_state})
        self.assertEqual(self.detector.sample(joint_state, action),
                         expected_obz)
        self.assertEqual(self.detector.probability(expected_obz,
                                                   joint_state, action), 1.0)


def tt(s2, s4, s6):
    """For convenience"""
    return {"s2": LocObjState(2, "obj2", {"loc": (s2,)}),
            "s4": LocObjState(4, "obj4", {"loc": (s4,)}),
            "s6": LocObjState(6, "obj6", {"loc": (s6,)})}

class TestCorrModel(unittest.TestCase):

    def setUp(self):
        self.robot_id = 0
        self.detector = EvenObjDetector(100, self.robot_id, radius=2)
        self.target_id = 2
        self.LEN = 5
        variables = ["s2", "s4", "s6"]
        weights = []
        for l2 in range(self.LEN):
            for l4 in range(self.LEN):
                for l6 in range(self.LEN):
                    s2 = LocObjState(2, "obj2", {"loc": (l2,)})
                    s4 = LocObjState(4, "obj4", {"loc": (l4,)})
                    s6 = LocObjState(6, "obj6", {"loc": (l6,)})
                    # object 2 and 4 are close, but 2 and 6 are far
                    if abs(l4 - l2) <= 1 and abs(l6 - l2) >= 1:
                        weights.append(((s2, s4, s6), 1.0))
                    else:
                        weights.append(((s2, s4, s6), 0.0))
        self.dist = TabularDistribution(variables, weights)
        self.corr_detector = CorrDetectorModel(self.target_id,
                                               {2, 4, 6},
                                               self.detector,
                                               self.dist)

    def test_sample(self):
        target_state = LocObjState(self.target_id, "obj2", {"loc": (1,)})
        robot_state = RobotState(self.robot_id, {"loc": (0,)})
        action = UseDetector(self.detector.id)
        joint_state = JointState({target_state.id: target_state,
                                  robot_state.id: robot_state})
        for i in range(100):
            z = self.corr_detector.sample(joint_state, action)
            self.assertEqual(z[self.target_id]["loc"], target_state["loc"])
            # The sampled observation contain null or, if not should
            # follow the distribution
            if not isinstance(z[4], NullObz):
                d = euclidean_dist(z[4]["loc"], z[self.target_id]["loc"])
                self.assertLessEqual(d, 1)
            if not isinstance(z[6], NullObz):
                d = euclidean_dist(z[6]["loc"], z[self.target_id]["loc"])
                self.assertGreaterEqual(d, 1)


    def test_prob(self):
        target_state = LocObjState(self.target_id, "obj2", {"loc": (1,)})
        robot_state = RobotState(self.robot_id, {"loc": (0,)})
        action = UseDetector(self.detector.id)
        joint_state = JointState({target_state.id: target_state,
                                  robot_state.id: robot_state})

        z1 = JointObz({
            2: ObjectObz(2, "obj2", {"loc": (3,)}),
            4: ObjectObz(4, "obj4", {"loc": (3,)}),
            6: ObjectObz(6, "obj6", {"loc": (3,)})
        })
        # Should be 0.0 because observation about target (obj2) mismatch the state
        self.assertEqual(self.corr_detector.probability(z1, joint_state, action),
                         0.0)

        z2 = JointObz({
            2: ObjectObz(2, "obj2", {"loc": (1,)}),
            4: ObjectObz(4, "obj4", {"loc": (0,)}),
            6: ObjectObz(6, "obj6", {"loc": (3,)})
        })
        # Should be 0.0 because, object 6 is outside of the robot's range
        self.assertEqual(self.corr_detector.probability(z2, joint_state, action),
                         0.0)

        z3 = JointObz({
            2: ObjectObz(2, "obj2", {"loc": (1,)}),
            4: ObjectObz(4, "obj4", {"loc": (0,)}),
            6: ObjectObz(6, "obj6", {"loc": (2,)})
        })
        # Should be 1.0 because it checks out
        self.assertEqual(self.corr_detector.probability(z3, joint_state, action),
                         1.0)

        z4 = JointObz({
            2: ObjectObz(2, "obj2", {"loc": (1,)}),
            4: NullObz(4),
            6: ObjectObz(6, "obj6", {"loc": (2,)})
        })
        # Should be probable, and greater than the probability for the case before, because we
        # do not know 4 and 4 can take more than one possible values
        self.assertGreater(self.corr_detector.probability(z4, joint_state, action),
                           self.corr_detector.probability(z3, joint_state, action))



if __name__ == "__main__":
    unittest.main()
