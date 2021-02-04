import copy

class Object:
    """
    An object is just an id. So, it is EQUIVALENT to
    the id, that is, an integer
    """
    def __init__(self, objid):
        self.id = objid

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if isinstance(other, int):
            return self.id == other
        elif isinstance(other, Object):
            return self.id == other.id
        else:
            return False


class ObjectSetting:
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
        if isinstance(other, ObjectSetting):
            return self.objclass == other.objclass\
                and self.attributes == other.attributes
        else:
            return False

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
        raise NotImplementedError("ObjectSetting is immutable.")

    def __len__(self):
        return len(self.attributes)

    def copy(self):
        """copy(self)
        Copies the state.
        You should override this method for efficiency,
        if deepcopy is not necessary"""
        return ObjectSetting(self.objid,
                             self.objclass,
                             copy.deepcopy(self.attributes))

    def get(self, attr, default_val):
        if attr in self.attributes:
            return self.attributes[attr]
        else:
            return default_val


class JointSetting:
    """
    State that can be factored by objects, that is, to ObjectState(s).

    __init__(self, object_states)
    """

    def __init__(self, object_settings):
        """
        Args:
            objects_settings (dict, or array-like): dictionary {ID:ObjectSetting},
                or an array like object consisting of ObjectSettings
        """
        if type(object_settings) != dict:
            # expect object_settings to be enumerable as an array
            object_settings_dict = {sobj.id : sobj
                                  for sobj in object_settings}
            # Check id uniqueness
            assert len(object_settings_dict) == len(object_settings),\
                "object_settings contains duplicated id"
            object_settings = object_settings_dict
        else:
            # Check integrity of object id in dict
            for objid in object_settings:
                assert object_settings[objid].id == objid,\
                    "Object setting id mismatch (%d != %d)" % (objid, object_settings[objid].id)

        self.object_settings = object_settings
        self._situation = frozenset(self.object_settings.items())
        self._hashcode = hash(self._situation)

    def __str__(self):
        return pprint.pformat(self.object_settings,
                              indent=2)

    def __repr__(self):
        return '%s::[%s]' % (str(self.__class__.__name__),
                             repr(self.object_settings))

    def __eq__(self, other):
        return isinstance(other, JointSetting)\
            and self.object_settings == other.object_settings

    def __hash__(self):
        return self._hashcode

    def obj(self, objid):
        return self.get_object_setting(objid)

    def __getitem__(self, objid):
        """__getitem__(self, attr)
        Returns the attribute"""
        return self.object_settings[objid]

    def __setitem__(self, attr, value):
        """__setitem__(self, attr, value)
        Sets the attribute `attr` to the given value."""
        raise NotImplementedError("ObjectSetting is immutable.")

    def copy(self):
        """copy(self)
        Copies the state.
        You should override this method for efficiency,
        if deepcopy is not necessary"""
        return JointSetting(copy.deepcopy(self.object_settings))

    def __len__(self):
        return len(self.object_settings)

    def __iter__(self):
        """Iterate over object ids"""
        return iter(self.object_settings)
