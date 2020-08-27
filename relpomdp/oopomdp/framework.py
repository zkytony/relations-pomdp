# Object Oriented POMDP framework - without object independence assumption

from pomdp_py.framework.basics import POMDP, State, Action, Observation,\
    ObservationModel, TransitionModel, GenerativeDistribution, Environment, Agent
import copy
from relpomdp.oopomdp.graph import *

########### State ###########
class ObjectState(State):
    # Note: 08/22/2020 - it's a copy of the ObjectState from pomdp_py
    """
    This is the result of OOState factoring; A state
    in an OO-POMDP is made up of ObjectState(s), each with
    an `object class` (str) and a set of `attributes` (dict).
    """
    def __init__(self, objclass, attributes):
        """
        class: "class",
        attributes: {
            "attr1": value,  # value should be hashable
            ...
        }
        """
        self.objclass = objclass
        self.attributes = attributes
        self._hashcode = hash(frozenset(self.attributes.items()))

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return '%s::(%s,%s)' % (str(self.__class__.__name__),
                                str(self.objclass),
                                str(self.attributes))
    
    def __hash__(self):
        return self._hashcode

    def __eq__(self, other):
        return self.objclass == other.objclass\
            and self.attributes == other.attributes

    def __getitem__(self, attr):
        """__getitem__(self, attr)
        Returns the attribute"""
        return self.attributes[attr]

    def __setitem__(self, attr, value):
        """__setitem__(self, attr, value)
        Sets the attribute `attr` to the given value."""
        self.attributes[attr] = value
    
    def __len__(self):
        return len(self.attributes)

    def copy(self):
        """copy(self)
        Copies the state."""
        raise NotImplementedError


class OOState(State):

    """
    State that can be factored by objects, that is, to ObjectState(s).

    __init__(self, object_states)
    """

    def __init__(self, object_states):
        """
        objects_states: dictionary of dictionaries; Each dictionary represents an object state:
            { ID: ObjectState }
        """
        # internally, objects are sorted by ID.
        self.object_states = object_states
        self._situation = frozenset(self.object_states.items())
        self._hashcode = hash(self._situation)

    @property
    def situation(self):
        """situation(self)
        This is a `frozenset` which can be used to identify
        the situation of this state since it supports hashing."""
        return self._situation

    def __str__(self):
        return '%s::[%s]' % (str(self.__class__.__name__),
                             str(self.object_states))

    def __repr__(self):
        return self.__str__()
    
    def __eq__(self, other):
        return isinstance(other, OOState)\
            and self.object_states == other.object_states

    def __hash__(self):
        return self._hashcode
    
    def get_object_state(self, objid):
        """get_object_state(self, objid)
        Returns the ObjectState for given object."""
        return self.object_states[objid]

    def set_object_state(self, objid, object_state):
        """set_object_state(self, objid, object_state)
        Sets the state of the given
        object to be the given object state (ObjectState)
        """        
        self.object_states[objid] = object_state

    def get_object_class(self, objid):
        """get_object_class(self, objid)
        Returns the class of requested object"""
        return self.object_states[objid].objclass

    def get_object_attribute(self, objid, attr):
        """get_object_attribute(self, objid, attr)
        Returns the attributes of requested object"""        
        return self.object_states[objid][attr]

    def copy(self):
        """copy(self)
        Copies the state."""
        raise NotImplementedError

    def obj(self, objid):
        return self.get_object_state(objid)

    def __getitem__(self, objid):
        return self.get_object_state(objid)
    
    def __setitem(self, objid, object_state):
        self.set_object_state(objid, object_state)
    
    def __len__(self):
        return len(self.object_states)

########### Observation ###########
class ObjectObservation(Observation):
    def __init__(self, objclass, attributes):
        self.objclass = objclass
        self.attributes = attributes
        self._hashcode = hash(frozenset(self.attributes.items()))

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return '%s::(%s,%s)' % (str(self.__class__.__name__),
                                str(self.objclass),
                                str(self.attributes))
    
    def __hash__(self):
        return self._hashcode

    def __eq__(self, other):
        return self.objclass == other.objclass\
            and self.attributes == other.attributes

    def __getitem__(self, attr):
        """__getitem__(self, attr)
        Returns the attribute"""
        return self.attributes[attr]

    def __setitem__(self, attr, value):
        """__setitem__(self, attr, value)
        Sets the attribute `attr` to the given value."""
        self.attributes[attr] = value
    
    def __len__(self):
        return len(self.attributes)

    def copy(self):
        """copy(self)
        Copies the state."""
        raise NotImplementedError

class OOObservation(Observation):
    def __init__(self, object_observations):
        """
        objects_observations: dictionary of dictionaries; Each dictionary represents an object observation:
            { ID: ObjectObservation }
        """
        # internally, objects are sorted by ID.
        self.object_observations = object_observations
        self._situation = frozenset(self.object_observations.items())
        self._hashcode = hash(self._situation)

    @property
    def situation(self):
        """situation(self)
        This is a `frozenset` which can be used to identify
        the situation of this observation since it supports hashing."""
        return self._situation

    def __str__(self):
        return '%s::[%s]' % (str(self.__class__.__name__),
                             str(self.object_observations))

    def __repr__(self):
        return self.__str__()
    
    def __eq__(self, other):
        return isinstance(other, OOObservation)\
            and self.object_observations == other.object_observations

    def __hash__(self):
        return self._hashcode
    
    def get_object_observation(self, objid):
        """get_object_observation(self, objid)
        Returns the ObjectObservation for given object."""
        return self.object_observations[objid]

    def set_object_observation(self, objid, object_observation):
        """set_object_observation(self, objid, object_observation)
        Sets the observation of the given
        object to be the given object observation (ObjectObservation)
        """        
        self.object_observations[objid] = object_observation

    def get_object_class(self, objid):
        """get_object_class(self, objid)
        Returns the class of requested object"""
        return self.object_observations[objid].objclass

    def get_object_attribute(self, objid, attr):
        """get_object_attribute(self, objid, attr)
        Returns the attributes of requested object"""        
        return self.object_observations[objid][attr]

    def copy(self):
        """copy(self)
        Copies the observation."""
        raise NotImplementedError

    def obj(self, objid):
        return self.get_object_observation(objid)

    def __getitem__(self, objid):
        return self.get_object_observation(objid)
    
    def __setitem(self, objid, object_observation):
        self.set_object_observation(objid, object_observation)
    
    def __len__(self):
        return len(self.object_observations)    


########### Condition and Effect ###########
class Condition:
    """Deterministic condition"""
    def satisfy(self, state, action):
        """Returns T/F based on whether the state satisfies the condition.
        In the case of Observation model, `state` should be interpreted
        as 'next_state'"""
        raise NotImplementedError
    
class TEffect(GenerativeDistribution):
    """Probabilistic transition effect"""
    def __init__(self, effect_type):
        self.effect_type = effect_type

    def random(self, state, action, byproduct=None):
        """Returns an OOState after applying this effect on `state`"""
        raise NotImplementedError

    def mpe(self, state, action, byproduct=None):
        """Returns an OOState after applying this effect on `state`"""
        raise NotImplementedError    

    def probability(self, next_state, state, action, byproduct=None):
        """Returns the probability of getting `next_state` if applying
        this effect on `state` given `action`."""
        raise NotImplementedError    

class DeterministicTEffect(TEffect):
    """Deterministically move"""
    def __init__(self, effect_type, epsilon=1e-9):
        self.epsilon = epsilon
        super().__init__(effect_type)
        
    def random(self, state, action, byproduct=None):
        """Returns an OOState after applying this effect on `state`"""
        return self.mpe(state, action, byproduct)

    def probability(self, next_state, state, action, byproduct=None):
        """Returns the probability of getting `next_state` if applying
        this effect on `state` given `action`."""
        expected_next_state = self.mpe(state, action)
        if next_state == expected_next_state:
            return 1.0 - self.epsilon
        else:
            return self.epsilon

class OEffect(GenerativeDistribution):
    """Probabilistic transition effect"""
    def __init__(self, effect_type):
        self.effect_type = effect_type

    def random(self, next_state, action, byproduct=None):
        """Returns an OOObservation after applying this effect on `state`"""
        raise NotImplementedError

    def mpe(self, next_state, action, byproduct=None):
        """Returns an OOObservation after applying this effect on `state`"""
        raise NotImplementedError    

    def probability(self, observation, next_state, action, byproduct=None):
        """Returns the probability of getting `observation` if applying
        this effect on `state` given `action`."""
        raise NotImplementedError

class DeterministicOEffect(OEffect):
    """Probabilistic transition effect"""
    def __init__(self, effect_type, epsilon=1e-9):
        self.epsilon = epsilon
        super().__init__(effect_type)

    def random(self, next_state, action, byproduct=None):
        """Returns an OOObservation after applying this effect on `state`"""
        return self.mpe(next_state, action, byproduct)

    def probability(self, observation, next_state, action, byproduct=None):
        """Returns the probability of getting `observation` if applying
        this effect on `state` given `action`."""
        expected_observation = self.mpe(state, action)
        if observation == expected_observation:
            return 1.0 - self.epsilon
        else:
            return self.epsilon
        
########### Class and Relation ###########    
class Class(Node):
    """A node, which could have an (x,y) location"""
    def __init__(self, name):
        """
        The name is expected to be unique in the graph.
        """
        super().__init__(name, data=name)

    @property
    def name(self):
        return self.data

    def __repr__(self):
        return "%s(%d)" % (type(self).__name__, self.id)

    def __hash__(self):
        return hash(self.id)

    
class Relation(Edge):
    """
    A relation is a directed edge.
    """
    def __init__(self, name, class1, class2):
        """class1 and class2 are expected to be Class objects (i.e. nodes).
        If they are strings, they will be converted to Class objects.
        The tuple (name, class1, class2) is expected to be unique.
        """
        if type(class1) == str:
            class1 = Class(class1)
        if type(class2) == str:
            class2 = Class(class2)
        self.name = name
        super().__init__((name, class1.name, class2.name),
                         class1, class2, data=name)

    def eval(self, object_state1, object_state2):
        """Returns True if the Relation holds. False otherwise"""
        raise NotImplementedError

    def __call__(self, object_state1, object_state2):
        """Returns True if the Relation holds. False otherwise"""
        return self.eval(object_state1, object_state2)

    @property
    def degenerate(self):
        return len(self.nodes) == 1

    @property
    def class1(self):
        return self.nodes[0]

    @property
    def class2(self):
        if len(self.nodes) >= 2:
            return self.nodes[1]
        else:
            return None
            
    def __repr__(self):
        if self.data is None:
            data = "--"
        else:
            data = self.data
        if not self.degenerate:
            return "%s(%s,%s)" % (self.name, self.class1.name, self.class2.name)
        else:
            return "%s(%s,%s)" % (self.name, self.class1.name)

    @property
    def color(self):
        return "black"

class InfoRelation(Relation):
    def to_mrf(self, *args, **kwargs):
        pass

class RelationGraph(Graph):
    def __init__(self, relations):
        super().__init__(relations, directed=True)
        
########### Object-Oriented Transition Model ###########        
class OOTransitionModel(TransitionModel):
    def __init__(self, cond_effects):
        self._cond_effects = cond_effects

    def _satisfied_effects(self, state, action):
        effects = []
        for condition, effect in self._cond_effects:
            res = condition.satisfy(state, action)
            if type(res) == tuple:
                satisfied, byproduct = res
            else:
                satisfied = res
                byproduct = None
            if satisfied:
                effects.append((effect, byproduct))
        return effects

    def sample(self, state, action, argmax=False):
        """sample(self, state, action, **kwargs)
        Samples the next state by applying effects with satisfying cond_effects
        """
        effects = self._satisfied_effects(state, action)
        next_state = state.copy()
        for effect, byproduct in effects:
            if argmax:
                next_state = effect.mpe(next_state, action, byproduct)
            else:
                next_state = effect.random(next_state, action, byproduct)
        return next_state
        
    def probability(self, next_state, state, action, **kwargs):
        """
        probability(self, next_state, state, action, **kwargs)
        Returns the probability of :math:`\Pr(s'|s,a)`.

        Here we assume the effects happen independently from each other;
        This is still -- a solution -- not part of the problem definition (as M.Littman would say?)
        TODO: What if effects are not independent?
        """
        effects = self._satisfied_effects(state, action)        
        prob = 1.0
        for effect, byproduct in effects:
            prob *= effect.probability(next_state, state, action, byproduct)
        return prob
    
    def argmax(self, state, action):
        """
        argmax(self, state, action, **kwargs)
        Returns the most likely next state"""
        return self.sample(state, action, argmax=True)
    

########### Object-Oriented Observation Model ###########            
class OOObservationModel(ObservationModel):
    def __init__(self, cond_effects):
        self._cond_effects = cond_effects

    def _satisfied_effects(self, state, action):
        effects = []
        for condition, effect in self._cond_effects:
            res = condition.satisfy(state, action)
            if type(res) == tuple:
                satisfied, byproduct = res
            else:
                satisfied = res
                byproduct = None
            if satisfied:
                effects.append((effect, byproduct))
        return effects

    def sample(self, next_state, action, argmax=False):
        """sample(self, next_state, action, **kwargs)
        Samples the observation by applying effects with satisfying cond_effects
        """
        effects = self._satisfied_effects(next_state, action)
        next_state = state.copy()
        for effect, byproduct in effects:
            if argmax:
                next_state = effect.mpe(next_state, action, byproduct)
            else:
                next_state = effect.random(next_state, action, byproduct)
        return next_state
        
    def probability(self, observation, next_state, action, **kwargs):
        """
        probability(self, next_state, state, action, **kwargs)
        Returns the probability of :math:`\Pr(s'|s,a)`.

        Here we assume the effects happen independently from each other;
        This is still -- a solution -- not part of the problem definition (as M.Littman would say?)
        TODO: What if effects are not independent?
        """
        effects = self._satisfied_effects(next_state, action)
        prob = 1.0
        for effect, byproduct in effects:
            prob *= effect.probability(observation, next_state, action, byproduct)
        return prob
    
    def argmax(self, state, action):
        """
        argmax(self, state, action, **kwargs)
        Returns the most likely next state"""
        return self.sample(state, action, argmax=True)


########### Object-Oriented Environment ###########        
class OOEnvironment(Environment):
    """
    OOEnvironment is initialized by both an initial state
    and a set of relations"""

    def __init__(self, init_state, relations, cond_effects, reward_model):
        """
        `relations`s is a set of relations that is relevant for this environment.

        `cond_effects` is a set of (Condition, Effects) pairs,
        which will be used to form the transition model of the OO-POMDP.
        """
        self._cond_effects = cond_effects
        self._relation_graph = RelationGraph(relations)
        transition_model = OOTransitionModel(cond_effects)
        super().__init__(init_state, transition_model, reward_model)

    
########### Object-Oriented Environment ###########        
class OOAgent(Agent):
    """
    OOAgent is initialized by both an initial state
    and a set of relations"""

    def __init__(self, init_belief,
                 cond_effects_t,
                 cond_effects_o,
                 policy_model, reward_model):
        """
        `relations`s is a set of relations that is relevant for this environment.

        `cond_effects` is a set of (Condition, Effects) pairs,
        which will be used to form the transition model of the OO-POMDP.
        """
        self._cond_effects_t = cond_effects_t
        transition_model = OOTransitionModel(cond_effects_t)
        self._cond_effects_o = cond_effects_o
        observation_model = OOTransitionModel(cond_effects_o)
        super().__init__(init_belief,
                         policy_model,
                         transition_model,
                         observation_model,
                         reward_model)
