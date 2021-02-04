import unittest
from corrsearch.models import *
from corrsearch.objects import *
from corrsearch.utils import *

# A toy detector model
class ToyDetector(DetectorModel):
    """This detector is only able to detect object with even id,
    and it detects them if robot is within a radius of them."""
    def __init__(self, detector_id, robot_id, radius=2):
        self.radius = 2
        super().__init__(detector_id, robot_id)

    def iprob(self, objobz, objstate, robot_state, action):
        """
        Returns the probability of Pr(zi | si, sr', a)
        """
        if objobz.objid % 2 == 0:
            if euclidean_dist(objstate.loc, robot_state.loc) <= self.radius:
                return indicator(objobz["loc"] == objstate.loc)
            else:
                return indicator(isinstance(objobz, NullObz))
        return 0.0

    def isample(self, objstate, robot_state, action):
        """
        Returns a sample zi according to Pr(zi | si, sr', a)
        """
        if objstate.objid % 2 == 0:
            if euclidean_dist(objstate.loc, robot_state.loc) <= self.radius:
                return ObjectObz(objstate.id,
                                 objstate.objclass,
                                 {"loc":objstate.loc})
        return NullObz(objstate.id)



class TestDetector(unittest.TestCase):
    def setUp(self):
        self.robot_id = 0
        self.detector = ToyDetector(1, self.robot_id, radius=2)

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



if __name__ == "__main__":
    unittest.main()
