from corrsearch.objects import ObjectState, JointState

class LocObjState(ObjectState):
    """State of an object that has 'location'
    as an attribute."""
    @property
    def loc(self):
        return self.attributes["loc"]

    def copy(self):
        return LocObjectState(self.objid,
                              self.objclass,
                              copy.deepcopy(self.attributes))

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
