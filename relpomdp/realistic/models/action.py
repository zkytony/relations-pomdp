# actions in Thor

import pomdp_py


class Action(pomdp_py.Action):
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


# Basic grid-based navigation actions
MoveAhead = Action("MoveAhead")
MoveBack = Action("MoveBack")
RotateLeft = Action("RotateLeft")
RotateRight = Action("RotateRight")
LookUp = Action("LookUp")
LookDown = Action("LookDown")

NAV_ACTIONS = {MoveAhead, MoveBack, RotateLeft, RotateRight, LookUp, LookDown}
