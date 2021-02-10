"""Object observation

Observation can be factored by objects.

For simple typing, will refer to observation by "Obz".

Defines ObjectObz and JointObz. They are very much the same
interface as ObjState and JointState.
"""
from pomdp_py.framework.basics import Observation
import copy
import pprint

class ObjectObz(Observation):
    def __init__(self, objid, objclass, attributes):
        """
        Args:
            objid (int): ID of the object
            class: "class",
            attributes: Dictionary, mapping from attribute name
                        to a HASHABLE attribute.
                {
                    "attr1": Attribute,
                    ...
                }.

        """
        self.id = objid
        self.objclass = objclass
        self.attributes = attributes
        self._hashcode = hash(frozenset(self.attributes.items()))

    def __repr__(self):
        return '{}::({},{},{})'.format(self.__class__.__name__,
                                       self.objid,
                                       self.objclass,
                                       self.attributes)

    def __str__(self):
        s = "{}#{} [{}]\n".format(self.objclass,
                                  self.objid,
                                  self.__class__.__name__)
        s += pprint.pformat(self.attributes, indent=2)
        return s

    def __hash__(self):
        return self._hashcode

    def __eq__(self, other):
        if isinstance(other, ObjectObz):
            return self.objclass == other.objclass\
                and self.attributes == other.attributes
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    @property
    def objid(self):
        return self.id

    def __getitem__(self, attr):
        """__getitem__(self, attr)
        Returns the attribute"""
        return self.attributes[attr]

    def __setitem__(self, attr, value):
        """__setitem__(self, attr, value)
        Sets the attribute `attr` to the given value."""
        raise NotImplementedError("ObjectObz is immutable.")

    def __len__(self):
        return len(self.attributes)

    def copy(self):
        """copy(self)
        Copies the observation.
        You should override this method for efficiency,
        if deepcopy is not necessary"""
        return ObjectObz(self.objid,
                             self.objclass,
                             copy.deepcopy(self.attributes))

    def get(self, attr, default_val):
        if attr in self.attributes:
            return self.attributes[attr]
        else:
            return default_val


class JointObz(Observation):
    """
    Observation that can be factored by objects, that is, to ObjectObz(s).

    __init__(self, object_obzs)
    """

    def __init__(self, object_obzs, label=None):
        """
        Args:
            objects_obzs (dict, or array-like): dictionary {ID:ObjectObz},
                or an array like object consisting of ObjectObzs
        """
        if type(object_obzs) != dict:
            # expect object_obzs to be enumerable as an array
            object_obzs_dict = {sobj.id : sobj
                                  for sobj in object_obzs}
            # Check id uniqueness
            assert len(object_obzs_dict) == len(object_obzs),\
                "object_obzs contains duplicated id"
            object_obzs = object_obzs_dict
        else:
            # Check integrity of object id in dict
            for objid in object_obzs:
                assert object_obzs[objid].id == objid,\
                    "Object obz id mismatch (%d != %d)" % (objid, object_obzs[objid].id)

        self.object_obzs = object_obzs
        self._situation = frozenset(self.object_obzs.items())
        self._hashcode = hash(self._situation)
        self._label = label

    def __str__(self):
        if self._label is not None:
            return self._label
        else:
            return pprint.pformat(self.object_obzs,
                                  indent=2)

    def __repr__(self):
        return '%s::[%s]' % (str(self.__class__.__name__),
                             repr(self.object_obzs))

    def __eq__(self, other):
        return isinstance(other, JointObz)\
            and self.object_obzs == other.object_obzs

    def __hash__(self):
        return self._hashcode

    def obj(self, objid):
        return self.get_object_obz(objid)

    def __getitem__(self, objid):
        """__getitem__(self, attr)
        Returns the attribute"""
        return self.object_obzs[objid]

    def __setitem__(self, attr, value):
        """__setitem__(self, attr, value)
        Sets the attribute `attr` to the given value."""
        raise NotImplementedError("ObjectObz is immutable.")

    def copy(self):
        """copy(self)
        Copies the obz.
        You should override this method for efficiency,
        if deepcopy is not necessary"""
        return JointObz(copy.deepcopy(self.object_obzs))

    def __len__(self):
        return len(self.object_obzs)

    def __iter__(self):
        """Iterate over object ids"""
        return iter(self.object_obzs)
