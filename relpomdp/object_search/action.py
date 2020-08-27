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

class Move(Action):
    EAST = (1, 0)  # x is horizontal; x+ is right. y is vertical; y+ is up.
    WEST = (-1, 0)
    NORTH = (0, 1)
    SOUTH = (0, -1)
    def __init__(self, name, motion):
        if motion not in {Move.EAST, Move.WEST,
                          Move.NORTH, Move.SOUTH}:
            raise ValueError("Invalid move motion %s" % motion)
        self.motion = motion
        super().__init__("move-%s" % name)

MoveE = Move("E", Move.EAST)
MoveW = Move("W", Move.WEST)
MoveN = Move("N", Move.NORTH)
MoveS = Move("S", Move.SOUTH)

class Pickup(Action):
    def __init__(self):
        super().__init__("pickup")
