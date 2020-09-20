from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.models import MarkovModel
from pgmpy.inference import BeliefPropagation
from pgmpy.sampling import GibbsSampling
import time


# Definition:
# value_index: The number used to represent a discrete variable's value in the pgmpy
# value_name: The name of that number
#   Example:
#   Say variable 'A' has three possible values, 'v1', 'v2', and 'v3'.
#   Then when you create a discreteFactor with 'A', pgmpy may map
#   0 to 'v1', 1 to 'v2', and 2 to 'v3'. You can also specify this
#   mapping through the "state_names" variable when creating the factor.

class SemanticMRF:
    def __init__(self, markov_model, value_to_name):
        """semantics_semantics (dict): map from tuple of integer indices (indicating variable values)
                to a meaningful value of the variable. Note that both the index and
                the value must be unique"""
        self.markov_model = markov_model

        self.value_to_name = value_to_name  # {variable -> {value_index -> value_name}}
        self.name_to_value = {}  # {variable -> {value_name -> value_index}}
        for variable in self.value_to_name:
            self.name_to_value[variable] =\
                {self.value_to_name[variable][value_index]:value_index
                 for value_index in range(len(self.value_to_name[variable]))}
        self.bp = BeliefPropagation(self.markov_model)
        # There's a warning from pgmpy at this line:
        #  "Found unknown state name. Trying to switch to using all state names as state numbers"
        # I have no clue why it happens. But it seems like the consequence is
        # the result of the sampling will be in value_index instead of value_name.
        # Therefore a conversion is needed in the "sample()" function below>.
        # self.gibbs = GibbsSampling(self.markov_model)  # Gibbs sampling is NOT NECESSARY for now.

    @property
    def G(self):
        return self.markov_model

    @property
    def factors(self):
        return self.G.factors

    def query(self, variables, evidence=None, verbose=False):
        """
        evidence is a mapping from variable to value_name. The value_name
            is the semantic one - e.g. for location, it's (x,y). Its
            integer index value in the MRF model will be used for
            actual inference.
        """
        for variable in variables:
            if not self.valid_var(variable):
                raise ValueError("Variable %s is not in the model" % variable)
        start_time = time.time()
        phi = self.bp.query(variables, evidence=evidence)
        if verbose:
            print("·····Query (%s|%s) took: %.5fs" % (str(variables), str(evidence),
                                                      time.time() - start_time))
        return phi

    def sample(self, evidence=None, size=1):
        samples = []
        for smpl in self.gibbs.generate_sample(size=size):
            semantic_sample = {}
            for varstate in smpl:
                variable = varstate.var
                value_index = varstate.state
                value_name = self.value_to_name[variable][value_index]
                semantic_sample[variable] = value_name
            samples.append(semantic_sample)
        return samples

    def valid_var(self, var):
        return var in set(self.G.nodes)

    def values(self, var):
        return list(self.name_to_value[var].keys())
    

def relations_to_mrf(relations):
    """Here, relations is a list of Relation objects"""    
    factors = []
    variables = []
    edges = []
    value_to_names = {}  # {variable -> {value_index -> value_name}}
    for relation in relations:
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
    return mrf
