import yaml
from relpomdp.pgm.mrf import SemanticMRF
from relpomdp.home2d.worlds.build_world import small_world1
from relpomdp.home2d.relation import Near
from relpomdp.home2d.state import *
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.models import MarkovModel
from pgmpy.inference import BeliefPropagation

# THIS TURNS OUT TO BE A LOT HARDER THAN I EXPECTED.
# The difficulty comes from generating multiple instances
# of objects of the same class. For example, two peppers
# and two salt. If you first generate peppers then generate
# salt, you need to use peppers as evidence. However the
# MRF is only at the class level (i.e. it cannot take
# two evidence of peppers at the same time). If you create
# one column per object then it's not scalable.
# FOR THIS REASON, I will hard code some example domains.
# Then for real, it should be tested on simulators like habitat.


class DomainCreator:
    """Creates an object search domain;
    The method of creating MRF could be reused in other tasks.
    But we assume we are solving object search right here.

    Example:

        Classes:
          Salt:
            loc: default

          Pepper:
            loc: default

          Robot:
            loc: top-right

        Relations:
          - near, Salt, Pepper

        Map: small_world1

    """
    def __init__(self):
        pass

    def read_file(self, domain_file, *args):
        with open(domain_file) as f:
            domain = yaml.load(f, Loader=yaml.Loader)
        return self.read(domain_file, *args)
            
    def read(self, domain,
             relation_name_dict,
             map_name_dict,
             relation_kwargs={}):
        """Reads a domain file, return an MRF and initial state of the environment

        relation_kwargs (dict): Supplies arguments to relation construction.
              maps from string name to a dictionary"""
        # Create the MRF relation graphical model
        grid_map = map_name_dict[domain["Map"]]()
        factors = []
        variables = []
        edges = []
        value_to_names = {}  # {variable -> {value_index -> value_name}}
        for relstr in domain["Relations"]:
            name, class1, class2 = map(str.strip, relstr.split(", "))
            relation = relation_name_dict[name](class1, class2,
                                                grid_map, **relation_kwargs[name])
            factor = relation.to_factor()  # MRF factor
            value_to_names.update(factor.no_to_name)
            variables.extend(factor.variables)
            edges.append([factor.variables[0], factor.variables[1]])
            factors.append(factor)
        G = MarkovModel()
        G.add_nodes_from(variables)
        G.add_edges_from(edges)
        G.add_factors(*factors)
        assert G.check_model()
        mrf = SemanticMRF(G, value_to_names)

        # Create environment
        config = mrf.sample()
        objstates = {}
        for class_name in domain["Classes"]:
            state = ItemState()
            
        

if __name__ == "__main__":
    filepath = "../tests/salt_pepper/domain.yaml"
    dc = DomainCreator()
    dc.read_file(filepath,
                 {"near": Near},
                 {"small_world1": small_world1})
            
