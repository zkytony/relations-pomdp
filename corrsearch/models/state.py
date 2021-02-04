from corrsearch.objects import ObjectState, JointState

class LocObjState(ObjectState):
    """State of an object that has 'location'
    as an attribute."""
    @property
    def loc(self):
        return self.attributes["loc"]

class RobotState(LocObjState):
    """
    Robot state is special
    """
    def __init__(self, objid, attributes, objclass="robot"):
        super().__init__(objid,
                         objclass,
                         attributes)
