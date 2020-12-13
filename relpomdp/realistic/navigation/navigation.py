# A (PO)MDP agent that can perform navigation on top of
# a THOR environment.

import pomdp_py as pdp
import math

# State
class NavState(pdp.State):
    def __init__(self, pos, rot):
        """
        pos: 2d position of robot, (x,z) in Unity coordinate system
        rot: a float for the rotation around y axis (vertical axis)
        """
        self.pos = pos
        self.rot = rot
    def __eq__(self, other):
        if isinstance(other, NavState):
            return other.pos == self.pos\
                and other.rot == self.rot
        return False
    def __hash__(self):
        return hash(self.pos, self.rot)

# Action
class Action(pdp.Action):
    """Mos action; Simple named action."""
    def __init__(self, name):
        self.name = name
    def __hash__(self):
        return hash(self.name)
    def __eq__(self, other):
        if isinstance(other, Action):
            return self.name == other.name
        elif type(other) == str:
            return self.name == other
    def __str__(self):
        return self.name
    def __repr__(self):
        return "Action(%s)" % self.name

class MotionAction(Action):
    def __init__(self, motion_name, motion):
        """
        motion (tuple): a (vt, vw) tuple for translationa, rotational velocities
        """
        self.motion = motion
        super().__init__(motion_name)

    def to_thor_action(self):
        """Returns an action representation in Thor
        as a tuple, (action_name:str, params:dict)"""
        vt, vw = self.motion
        if vt != 0 and vw != 0:
            raise ValueError("Ai2Thor does not support actions"\
                             "to change transition and rotation together.")
        if vt != 0:
            return (self.name, {"moveMagnitude": abs(vt)})
        else:
            return (self.name, {"degrees": abs(vw)})


def build_motion_actions(grid_size=0.25, degrees=45):
    # Ai2Thor motion actions
    return {MotionAction("MoveAhead", (grid_size, 0)),
            MotionAction("MoveBack", (-grid_size, 0)),
            MotionAction("RotateLeft", (0, -math.radians(degrees))),
            MotionAction("RotateRight", (0, math.radians(degrees)))}
