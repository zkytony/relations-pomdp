import pomdp_py
import corrsearch.objects

class SearchProblem:
    """
    A search problem. The problem is defined by
    a set of possible locations,
    a set of objects,
    a joint distribution of their locations,
    a target object.
    and a robot model:
    - a set of object detectors,
        each represented as a sensor observation model
    - a model of the robot state transition,
       (actions are MOVE, LOOK, DECLARE)
    """
    def __init__(self,
                 locations,
                 objects,
                 joint_dist,
                 target_object,
                 robot_model):
        self.locations      = locations
        self.objects        = objects
        self.joint_dist     = joint_dist
        self.target_object  = target_object
        self.robot_model    = robot_model

    def instantiate(self, *args, **kwargs):
        """This should return a search problem instance by creating a
        pomdp_py.Environment and an pomdp_py.Agent for it. We CHOOSE not to
        instantiate a custom Environment, but to just create one. Because an
        Environment always needs to have a true state, and an Agent needs to
        have a belief, but a search problem does not.
        """
        raise NotImplementedError

    def visualizer(self, *args, **kwargs):
        """Returns a visualizer of the problem"""
        raise NotImplementedError
