"""
How to specify a joint distribution?

That is not the question - I assume it can be done.
For example:
- Full table
- Graphical model (Bayesian network, Markov Random Field)
- Sum Product Networks
- Generative function

Also, you don't need to go too above and beyond - What
we care about is a joint distribution of object locations.
Obviously, the distribution does not need to be specified
at the object-level. It could be class level. How to use
the distribution should depend on the domain.
"""
class JointDist:
    """
    A JointDist represents a distribution over
    N variables, v1, ..., vN. How the distribution
    is represented, is up to the children class.
    """
    def __init__(self, variables):
        """
        Args:
            variables (array-like): List of references to variables.
                Could be string names, for example. A variable should
                be a hashable object
        """
        self._variables = variables

    def prob(self, values):
        """
        Args:
            values (dict): Mapping from variable to value.
                Does not have to specify the value for every variable
        """
        raise NotImplementedError

    def sample(self):
        raise NotImplementedError

    def marginal(self, variables, observation=None):
        """Performs marignal inference,
        produce a joint distribution over `variables`,
        given evidence (if supplied)"""
        raise NotImplementedError
