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


class MotionPolicy:
    def __init__(self, grid_map, all_motion_actions={MoveN, MoveS, MoveE, MoveW}):
        # Compute the valid motion actions at every location
        self.all_motion_actions = all_motion_actions
        self._legal_actions = {}  # maps from (x,y) to legal motion actions
        for x in range(grid_map.width):
            for y in range(grid_map.length):
                objstate = PoseState("PoseObject", {"pose":(x,y)})
                motion_actions = set(all_motion_actions)
                for wall_id in grid_map.walls:
                    wall = grid_map.walls[wall_id]
                    if touch_N(objstate, wall):
                        motion_actions.remove(MoveN)
                    elif touch_S(objstate, wall):
                        motion_actions.remove(MoveS)
                    elif touch_E(objstate, wall):
                        motion_actions.remove(MoveE)
                    elif touch_W(objstate, wall):
                        motion_actions.remove(MoveW)
                self._legal_actions[(x,y)] = motion_actions
        self._grid_map = grid_map

    def valid_motions(self, robot_pose):
        """
        Returns a set of MotionAction(s) that are valid to
        be executed from robot pose (i.e. they will not bump
        into obstacles). The validity is determined under
        the assumption that the robot dynamics is deterministic.
        """
        return set(self._legal_actions[robot_pose[:2]])

    def path_between(self, position1, position2, return_actions=True):
        """Note that for the return_actions=True case to return reasonable
        actions, the motion actions scheme needs to be `xy`, i.e. along the axes"""
        # Finds a path between position1 and position2.
        # Uses the Dijkstra's algorithm.
        V = set({(x,y)    # all valid positions
                 for x in range(self._grid_map.width) 
                 for y in range(self._grid_map.length)
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
            neighbors = self.get_neighbors(v)
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

    def get_neighbors(self, robot_pose):
        neighbors = {MoveEffect.move_by(robot_pose, action.motion):action
                     for action in self.valid_motions(robot_pose)}
        return neighbors


# class Pickup(Action):
#     def __init__(self):
#         super().__init__("pickup")


# Unittest
def unittest():
    # assert Pickup() == Pickup()
    assert MoveE == Move("E", Move.EAST)
    assert MoveW == Move("W", Move.WEST)
    assert MoveN == Move("N", Move.NORTH)
    assert MoveS == Move("S", Move.SOUTH)
    print("Passed.")

if __name__ == "__main__":
    unittest()
