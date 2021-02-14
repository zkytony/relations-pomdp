from corrsearch.objects import ObjectState, JointState
import copy

class LocObjState(ObjectState):
    """State of an object that has 'location'
    as an attribute."""
    @property
    def loc(self):
        return self.attributes["loc"]

    def copy(self):
        return LocObjState(self.objid,
                           self.objclass,
                           copy.deepcopy(self.attributes))

    def __lt__(self, other):
        return self.loc < other.loc

    def __hash__(self):
        return hash((self.objid, self.objclass, self.loc))


class RobotState(LocObjState):
    """
    Robot state is special
    """
    def __init__(self, objid, attributes, objclass="robot"):
        super().__init__(objid,
                         objclass,
                         attributes)

    def copy(self):
        return RobotState(self.objid,
                          copy.deepcopy(self.attributes),
                          objclass=self.objclass)

    @property
    def pose(self):
        if "pose" in self.attributes:
            return self.attributes["pose"]
        else:
            return self.attributes["loc"]

    def __hash__(self):
        return hash((self.objid, self.objclass, self.pose))

    @property
    def terminal(self):
        return self.attributes["terminal"]
