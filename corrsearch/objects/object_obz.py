"""Object observation

Observation can be factored by objects.

For simple typing, will refer to observation by "Obz".

Defines ObjectObz and JointObz. They are very much the same
interface as ObjState and JointState.
"""
from pomdp_py.framework.basics import Observation
from corrsearch.objects.template import ObjectSetting, JointSetting
import copy

class ObjectObz(ObjectSetting, Observation):
    def __init__(self, objid, objclass, attributes):
        ObjectSetting.__init__(self, objid, objclass, attributes)

    def copy(self):
        """copy(self)
        Copies the state.
        You should override this method for efficiency,
        if deepcopy is not necessary"""
        return ObjectObz(self.objid,
                         self.objclass,
                         copy.deepcopy(self.attributes))

class JointObz(JointSetting, Observation):
    """
    Observation that can be factored by objects, that is, to ObjectObz(s).

    __init__(self, object_obzs)
    """

    def __init__(self, object_obzs):
        """
        Args:
            objects_obzs (dict, or array-like): dictionary {ID:ObjectObz},
                or an array like object consisting of ObjectObzs
        """
        JointSetting.__init__(self, object_obzs)

    def copy(self):
        """You should override this method for efficiency,"""
        object_obzs = {objid : self.object_obzs[objid].copy()
                       for objid in self.object_obzs}
        return JointObz(object_obzs)

    @property
    def object_obzs(self):
        return self.object_settings
