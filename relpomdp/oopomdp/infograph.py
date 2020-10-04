# Relational Graph over attributes

from relpomdp.oopomdp.graph import *
from pgmpy.factors.discrete import DiscreteFactor

class Domain:
    def __contains__(self, value):
        raise NotImplementedError

    def values(self):
        raise NotImplementedError

class Attribute(Node):

    """An attribute is specified by a class
    and an attribute name. It can be grounded
    into multiple instances. Each grounding
    is a tuple (value, Domain)"""

    def __init__(self, _class, name):
        self._class = _class
        self.name = name
        self.groundings = []
        super().__init__("%s-%s" % (_class, name))

    @property
    def clas(self):
        return self._class

    def ground(self, value, domain):
        self.groundings.append((value, domain))

class InfoRelation(Edge):
    """
    An InfoRelation is a directed edge defined
    by two attributes, and a name. It can be
    grounded as (Domain1, Domain2, func), where
    Domain1 and Domain2 are the domains of
    groundings of the two attributes, respectively,
    and func a function that maps x1,x2 to a probability
    (essentially pairwise potential)
    """
    def __init__(self, attr1, attr2, name):
        super().__init__("%s:%s:%s" % (attr1.id, name, attr2.id),
                         attr1, attr2)
        self.grounding = None
        self.potential = None

    @property
    def attr1(self):
        return self.nodes[0]
    
    @property
    def attr2(self):
        return self.nodes[1]

    def ground(self, domain1, domain2, func):
        self.grounding = (domain1, domain2, func)

    def grounding_to_potential(self):
        d1, d2, func = self.grounding
        values1 = d1.values()
        values2 = d2.values()
        card1 = len(values1)
        card2 = len(values2)
        variables = [self.attr1.id, self.attr2.id]
        edges = [[variables[0], variables[1]]]
        potentials = []
        value_names = {
            variables[0]: list(values1),
            variables[1]: list(values2)
        }
        for i, val_i in enumerate(value_names[variables[0]]):
            for j, val_j in enumerate(value_names[variables[1]]):
                potential = func(val_i, val_j)
                potentials.append(potential)
        self.potential = DiscreteFactor(variables, cardinality=[card1, card2],
                                        values=potentials, state_names=value_names)
        return self.potential


class RelationGraph(Graph):
    def __init__(self, relations):
        super().__init__(relations, directed=True)
        
