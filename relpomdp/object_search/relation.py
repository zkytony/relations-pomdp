import relpomdp.oopomdp.framework as oopomdp
from relpomdp.object_search.state import WallState

# Relations
class Touch(oopomdp.Relation):
    """Touching a wall"""
    def __init__(self, direction, class1, class2):
        if direction not in {"N", "E", "S", "W"}:
            raise ValueError("Invalid direction %s" % direction)
        self.direction = direction
        super().__init__("touch-%s" % direction,
                         class1, class2)
        
    def eval(self, object_state1, object_state2):
        """Returns True if the Relation holds. False otherwise.
        According to the paper, touch(o1,o2) holds if
        o2 is exactly one cell North, South, East, or West of o1.

        Note that we assume a vertical wall is on the right edge
        of a grid cell and a horizontal wall is on the top edge.
        Touching a wall means having a wall on any of the edges
        that the object_state1 is in."""
        if not isinstance(object_state2, WallState):
            raise ValueError("Taxi domain (at least in OO-MDP) requires"
                             "the Touch relation to involve Wall as the second object.")
        x1, y1 = object_state1.pose
        x2, y2 = object_state2.pose
        if self.direction == "N":
            if object_state2.direction == "H":
                return x1 == x2 and y1 == y2
            else:
                return False  # vertical wall cannot touch at North
        elif self.direction == "S":
            if object_state2.direction == "H":
                return x1 == x2 and y1 == y2 + 1
            else:
                return False  # vertical wall cannot touch at North
        elif self.direction == "E":
            if object_state2.direction == "V":
                return x1 == x2 and y1 == y2
            else:
                return False  # vertical wall cannot touch at East
        else:
            assert self.direction == "W"
            if object_state2.direction == "V":
                return x1 == x2 + 1 and y1 == y2
            else:
                return False  # vertical wall cannot touch at East

touch_N = Touch("N", "PoseObject", "Wall")
touch_S = Touch("S", "PoseObject", "Wall")
touch_E = Touch("E", "PoseObject", "Wall")
touch_W = Touch("W", "PoseObject", "Wall")

class On(oopomdp.Relation):
    def __init__(self, class1, class2):
        super().__init__("on",
                         class1, class2)
        
    def eval(self, object_state1, object_state2):
        """Returns True if the Relation holds. False otherwise.
        According to the paper, on(o1,o2) holds if o1 and o2 are
        overlapping"""
        return object_state1.pose == object_state2.pose            

is_on = On("PoseObject", "PoseObject")

# Extensions: Relations that are used not for transiitions
