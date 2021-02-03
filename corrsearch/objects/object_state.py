"""Objects state

An object can be literally represented just by a number, its ID.

What is more meaningful is the state of the object, ObjectState.

This has been defined in the OOPOMDP framework, but I will
port over the ObjectState class here, specifically.

It is, perhaps, the same framework as OOPOMDP. So, probably I
should reuse the code there. That is the right thing to do.
"""
from pomdp_py.framework.basics import State
from template import ObjectSetting, JointSetting
import pprint
import copy

class ObjectState(ObjectSetting, State):
    # Note: 08/22/2020 - it's a copy of the ObjectState from pomdp_py
    """
    This is the result of JointState factoring; A state
    in an OO-POMDP is made up of ObjectState(s), each with
    an `object class` (str) and a set of `attributes` (dict).
    """
    def __init__(self, objid, objclass, attributes):
        ObjectSetting.__init__(self, objid, objclass, attributes)

    def copy(self):
        """copy(self)
        Copies the state.
        You should override this method for efficiency,
        if deepcopy is not necessary"""
        return ObjectState(self.objid,
                           self.objclass,
                           copy.deepcopy(self.attributes))

class JointState(JointSetting, State):
    """
    State that can be factored by objects, that is, to ObjectState(s).

    __init__(self, object_states)
    """

    def __init__(self, object_states):
        """
        Args:
            objects_states (dict, or array-like): dictionary {ID:ObjectState},
                or an array like object consisting of ObjectStates
        """
        JointSetting.__init__(self, object_states)

    @property
    def object_states(self):
        return self.object_settings

    def copy(self):
        """You should override this method for efficiency,"""
        object_states = {objid : self.object_states[objid].copy()
                         for objid in self.object_states}
        return JointState(object_states)
