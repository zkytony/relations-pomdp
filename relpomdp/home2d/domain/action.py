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


def path_between(position1, position2, grid_map, get_neighbors_func,
                 return_actions=True):
    """Note that for the return_actions=True case to return reasonable
    actions, the motion actions scheme needs to be `xy`, i.e. along the axes"""
    # Finds a path between position1 and position2.
    # Uses the Dijkstra's algorithm.
    V = set({(x,y)    # all valid positions
             for x in range(width)
             for y in range(length)
             if self._grid_map.within_bounds((x,y))})
    position1 = position1[:2]  # If it is robot pose then it has length 3.
    S = set({})
    d = {v:float("inf")
         for v in V
         if v != position1}
    d[position1] = 0
    prev = {position1: None}
    while len(S) < len(V):
        diff_set = V - S
        v = min(diff_set, key=lambda v: d[v])
        S.add(v)
        neighbors = get_neighbors_func(v)
        for w in neighbors:
            motion_action = neighbors[w]
            cost_vw = motion_action.distance_cost
            if d[v] + cost_vw < d[w[:2]]:
                d[w[:2]] = d[v] + cost_vw
                prev[w[:2]] = (v, motion_action)

    # Return a path
    path = []
    pair = prev[position2[:2]]
    if pair is None:
        if not return_actions:
            path.append(position2)
    else:
        while pair is not None:
            position, action = pair
            if return_actions:
                # Return a sequence of actions that moves the robot from position1 to position2.
                path.append(action)
            else:
                # Returns the grid cells along the path
                path.append(position)
            pair = prev[position]
    return list(reversed(path))



# Unittest
def unittest():
    assert MoveE == Move("E", Move.EAST)
    assert MoveW == Move("W", Move.WEST)
    assert MoveN == Move("N", Move.NORTH)
    assert MoveS == Move("S", Move.SOUTH)
    print("Passed.")

if __name__ == "__main__":
    unittest()
