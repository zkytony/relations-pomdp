import pomdp_py
import math

# Primitive actions that the robot can perform in a home2d task

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
    EAST = (1, 0, 0)  # x is horizontal; x+ is right. y is vertical; y+ is up.
    WEST = (-1, 0, math.pi)
    NORTH = (0, 1, math.pi/2)
    SOUTH = (0, -1, 3*math.pi/2)
    def __init__(self, name, motion, distance_cost=1):
        if motion not in {Move.EAST, Move.WEST,
                          Move.NORTH, Move.SOUTH}:
            raise ValueError("Invalid move motion %s" % motion)
        self.motion = motion
        self.distance_cost = distance_cost
        super().__init__("move-%s" % name)

MoveE = Move("E", Move.EAST)
MoveW = Move("W", Move.WEST)
MoveN = Move("N", Move.NORTH)
MoveS = Move("S", Move.SOUTH)

class Pickup(Action):
    def __init__(self):
        super().__init__("pickup")


# Unittest
def unittest():
    assert Pickup() == Pickup()
    assert MoveE == Move("E", Move.EAST)
    assert MoveW == Move("W", Move.WEST)
    assert MoveN == Move("N", Move.NORTH)
    assert MoveS == Move("S", Move.SOUTH)
    print("Passed.")

if __name__ == "__main__":
    unittest()
