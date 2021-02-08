from corrsearch.objects import ObjectState, JointState
from corrsearch.models.state import LocObjState, RobotState

class F2DLocObjState(LocObjState):
    """State of an object that has 'location'
    as an attribute."""
    def __init__(self, objid, objclass, loc):
        super().__init__(objid, objclass, {"loc":loc})

    def copy(self):
        return F2DLocObjState(self.objid,
                              self.objclass,
                              {"loc": self["loc"]})


class F2DRobotState(LocObjState):
    """
    Robot state is special
    """
    def __init__(self, objid, pose, energy, objclass="robot"):
        super().__init__(objid,
                         objclass,
                         {"loc": pose[:2],
                          "pose": pose,
                          "energy": energy})

    def copy(self):
        return F2DRobotState(self.objid,
                             {"loc": self["loc"],
                              "pose": self["pose"],
                              "energy": self["energy"]},
                              objclass=self.objclass)

    @property
    def pose(self):
        if "pose" in self.attributes:
            return self.attributes["pose"]
        else:
            return self.attributes["loc"]
