"""Objects state

An object can be literally represented just by a number, its ID.

What is more meaningful is the state of the object, ObjectState.

This has been defined in the OOPOMDP framework, but I will
port over the ObjectState class here, specifically.

It is, perhaps, the same framework as OOPOMDP. So, probably I
should reuse the code there. That is the right thing to do.
"""
import pomdp_py
from pomdp_py.framework.basics import State
import pprint
import copy

class ObjectState(State):
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
        if isinstance(other, ObjectState):
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
        raise NotImplementedError("ObjectState is immutable.")

    def __len__(self):
        return len(self.attributes)

    def copy(self):
        """copy(self)
        Copies the state.
        You should override this method for efficiency,
        if deepcopy is not necessary"""
        return ObjectState(self.objid,
                             self.objclass,
                             copy.deepcopy(self.attributes))

    def get(self, attr, default_val):
        if attr in self.attributes:
            return self.attributes[attr]
        else:
            return default_val


class JointState(State):
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
        if type(object_states) != dict:
            # expect object_states to be enumerable as an array
            object_states_dict = {sobj.id : sobj
                                  for sobj in object_states}
            # Check id uniqueness
            assert len(object_states_dict) == len(object_states),\
                "object_states contains duplicated id"
            object_states = object_states_dict
        else:
            # Check integrity of object id in dict
            for objid in object_states:
                assert object_states[objid].id == objid,\
                    "Object state id mismatch (%d != %d)" % (objid, object_states[objid].id)

        self.object_states = object_states
        self._situation = frozenset(self.object_states.items())
        self._hashcode = hash(self._situation)

    def __str__(self):
        return pprint.pformat(self.object_states,
                              indent=2)

    def __repr__(self):
        return '%s::[%s]' % (str(self.__class__.__name__),
                             repr(self.object_states))

    def __eq__(self, other):
        return isinstance(other, JointState)\
            and self.object_states == other.object_states

    def __hash__(self):
        return self._hashcode

    def obj(self, objid):
        return self.get_object_state(objid)

    def __getitem__(self, objid):
        """__getitem__(self, attr)
        Returns the attribute"""
        return self.object_states[objid]

    def __setitem__(self, attr, value):
        """__setitem__(self, attr, value)
        Sets the attribute `attr` to the given value."""
        raise NotImplementedError("ObjectState is immutable.")

    def copy(self):
        """copy(self)
        Copies the state.
        You should override this method for efficiency,
        if deepcopy is not necessary"""
        return JointState(copy.deepcopy(self.object_states))

    def __len__(self):
        return len(self.object_states)

    def __iter__(self):
        """Iterate over object ids"""
        return iter(self.object_states)


class JointBelief(pomdp_py.GenerativeDistribution):
    """
    Belief factored by objects.
    """
    def __init__(self, object_beliefs):
        """
        object_beliefs (objid -> GenerativeDistribution)
        """
        self._object_beliefs = object_beliefs

    def __getitem__(self, state):
        """__getitem__(self, state)
        Returns belief probability of given state"""
        if not isinstance(state, JointState):
            raise ValueError("state must be JointState")
        belief_prob = 1.0
        for objid in self._object_beliefs:
            object_state = state.object_states[objid]
            belief_prob = belief_prob * self._object_beliefs[objid].probability(object_state)
        return belief_prob

    def mpe(self, **kwargs):
        """mpe(self, **kwargs)
        Returns most likely state."""
        object_states = {}
        for objid in self._object_beliefs:
            object_states[objid] = self._object_beliefs[objid].mpe(**kwargs)
        return JointState(object_states)

    def random(self, **kwargs):
        """random(self, **kwargs)
        Returns a random state"""
        object_states = {}
        for objid in self._object_beliefs:
            object_states[objid] = self._object_beliefs[objid].random(**kwargs)
        return JointState(object_states)

    def __setitem__(self, oostate, value):
        """__setitem__(self, oostate, value)
        Sets the probability of a given `oostate` to `value`.
        Note always feasible."""
        raise NotImplemented

    def object_belief(self, objid):
        """object_belief(self, objid)
        Returns the belief (GenerativeDistribution) for the given object."""
        return self._object_beliefs[objid]

    def set_object_belief(self, objid, belief):
        """set_object_belief(self, objid, belief)
        Sets the belief of object to be the given `belief` (GenerativeDistribution)"""
        self._object_beliefs[objid] = belief

    @property
    def object_beliefs(self):
        """object_beliefs(self)"""
        return self._object_beliefs

    def obj(self, objid):
        return self._object_beliefs[objid]

    def update(self, observation, action):
        raise NotImplementedError
