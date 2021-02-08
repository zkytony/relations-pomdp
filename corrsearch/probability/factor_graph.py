"""
Factor graph representation
of joint distribution, using pgmpy.
Custom interface
"""
from pgmpy.models import FactorGraph as PGMFactorGraph
from pgmpy.factors.discrete import DiscreteFactor
import itertools
from corrsearch.probability.dist import JointDist

class FactorGraph(JointDist):

    def __init__(self,
                 variables,
                 factors):
        """
        Args:
            variables (list): List of variables
            factors (list): List of TabularDistribution representation of factors.
        """
        self.fg = PGMFactorGraph()
        self.fg.add_nodes_from(variables)
        # Create factors using pgmpy's interface
        pgmpy_factors = []
        edges = []
        for factor in factors:
            # cardinality
            cardinality = [len(factor.valrange(var)) for var in factor.variables]
            # obtain state names
            state_names = {}
            for var in factor.variables:
                state_names[var] = list(sorted(factor.valrange(var),
                                               key=lambda si: si["loc"]))

            values = []
            for setting in itertools.product(*[state_names[var]
                                               for var in factor.variables]):
                setting_dict = {factor.variables[i]:setting[i]
                                for i in range(len(factor.variables))}
                prob = factor.prob(setting_dict)
                values.append(prob)

            phi = DiscreteFactor(factor.variables,
                                 cardinality,
                                 values,
                                 state_names=state_names)
            pgmpy_factors.append(phi)
            for var in factor.variables:
                edges.append((var, phi))
        self.fg.add_factors(*pgmpy_factors)
        self.fg.add_edges_from(edges)
        self.fg.check_model()
