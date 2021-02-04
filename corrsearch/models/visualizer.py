"""
Visualizer
visualizes an instance of a search problem
by visualizing the state of the instance.
"""

class Visualizer:
    def __init__(self, problem):
        """
        Given a problem (SearchProblem), create
        a visualizer for it
        """
        self.problem = problem

    def visualize(self, state, belief=None):
        """Visualizes the given state,
        which should come from an instantiated
        version (i.e. an environment) of
        the search problem."""
        raise NotImplementedError
