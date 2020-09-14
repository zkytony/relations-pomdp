import relpomdp.oopomdp.framework as oopomdp
from relpomdp.pgm.mrf import SemanticMRF
from relpomdp.object_search.state import WallState
from relpomdp.object_search.utils import euclidean_dist
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.models import MarkovModel
from pgmpy.inference import BeliefPropagation
from pgmpy.sampling import GibbsSampling

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
        x1, y1 = object_state1.pose[:2]
        x2, y2 = object_state2.pose[:2]
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
        return object_state1.pose[:2] == object_state2.pose[:2]

is_on = On("PoseObject", "PoseObject")

# Extensions: Relations that are used not for transiitions
class Near(oopomdp.InfoRelation):
    def __init__(self,
                 class1, class2,
                 pose_attr="pose", negate=False):
        self.pose_attr = pose_attr
        self.negate = negate  # True if "not near"
        super().__init__("near",
                         class1, class2)
        
    def eval(self, object_state1, object_state2):
        """Returns True if the Relation holds. False otherwise.
        According to the paper, on(o1,o2) holds if o1 and o2 are
        overlapping"""
        if object_state1.objclass == self.class1.name\
           and object_state2.objclass == self.class2.name:
            return self.is_near(object_state1.pose[:2], object_state2.pose[:2])
        return False

    def to_factor(self, locations1, locations2, is_near_func):
        """Return a DiscreteFactor representation of the grounded factor"""
        card1 = len(locations1)
        card2 = len(locations2)
        variables = ["%s_%s" % (self.class1.name, self.pose_attr),
                     "%s_%s" % (self.class2.name, self.pose_attr)]
        edges = [[variables[0], variables[1]]]

        # potentials: a list of joint potentials for the table.
        potentials = []
        
        # value_names: pgmpy's discrete factor expects every value be
        # indexed by an integer. Here we are recording the actual meaning
        # of that integer (e.g. 12 could mean the location (4,14))
        value_names = {
            variables[0]: list(locations1),
            variables[1]: list(locations2),            
        }
        
        # semantics = {}  # tabular entry (i,j) to semantic meaning (val_i, val_j)
        # index_to_semantics = []
        for i, loc_i in enumerate(locations1):
            for j, loc_j in enumerate(locations2):
                # semantics[(i,j)] = (loc_i, loc_j)
                near = is_near_func(loc_i, loc_j)
                if not self.negate:
                    potential = 1.0-1e-9 if near else 1e-9
                else:
                    # negation
                    potential = 1e-9 if near else 1.0-1e-9
                potentials.append(potential)
                    
        factor = DiscreteFactor(variables, cardinality=[card, card],
                                values=potentials, state_names=value_names)
        return factor
        

class GroundLevelNear(Near):
    def __init__(self, class1, class2,
                 grid_map, pose_attr="pose", negate=False):
        self.grid_map = grid_map
        super().__init__(class1, class2,
                         pose_attr=pose_attr, negate=negate)
        
    def is_near(self, pose1, pose2):
        same_room = (self.grid_map.room_of(pose1)\
                         == self.grid_map.room_of(pose2))
        return same_room\
            and euclidean_dist(pose1, pose2) <= 2

    def to_factor(self):
        locations = {(x,y)
                     for x in range(self.grid_map.width)
                     for y in range(self.grid_map.length)}
        return super().to_factor(locations, locations, self.is_near)

class RoomLevelNear(Near):
    def __init__(self, class1, class2,
                 grid_map, pose_attr="pose", negate=False):
        self.grid_map = grid_map
        super().__init__(class1, class2,
                         pose_attr=pose_attr, negate=negate)
        
    def is_near(self, pose1, pose2):
        same_room = (self.grid_map.room_of(pose1)\
                         == self.grid_map.room_of(pose2))
        return same_room

    def to_factor(self):
        rooms = self.grid_map.containers("Room")
        locations = [name for name in rooms]
        return super().to_factor(locations, locations, self.is_near)    


class In(oopomdp.InfoRelation):
    def __init__(self,
                 item_class, container_class, container_type,
                 grid_map, pose_attr="pose", negate=False,
                 container_level=False):
        """
        Args:
            container_level (bool): If True, then the factor will be joining
                the item and container both at the granularity of the container.
        """
        self.grid_map = grid_map
        self.pose_attr = pose_attr
        self.negate = negate
        self.item_class = item_class
        self.container_class = container_class
        self.container_type = container_type
        self.container_level = container_level
        super().__init__("in",
                         item_class, container_class)

    def eval(self, item_state, container_state):
        """Returns True if the Relation holds. False otherwise.
        According to the paper, on(o1,o2) holds if o1 and o2 are
        overlapping"""
        return item_state.pose[:2] in container_state.footprint

    def is_in(self, pose, container):
        """container (ContainerState)"""
        if self.container_level:
            # pose is at container level (i.e. it is a container name)
            return pose == container.name
        else:
            return pose in container.footprint

    def to_factor(self):
        # For right now, we will assume the robot knows that the grid map
        # contains a fixed number of containers. So basically the factor
        # is an enumeration of those containers.
        containers = self.grid_map.containers(self.container_type)
        if self.container_level:
            locations = [name for name in containers]
        else:
            locations = [(x,y)
                         for x in range(self.grid_map.width)\
                         for y in range(self.grid_map.length)] 
        item_card = len(locations)
        container_card = len(containers)
        variables = ["%s_%s" % (self.item_class, self.pose_attr),
                     "%s" % (self.container_class)]
        edges = [[variables[0], variables[1]]]
        
        # potentials: a list of joint potentials for the table.
        potentials = []

        # value_names: pgmpy's discrete factor expects every value be
        # indexed by an integer. Here we are recording the actual meaning
        # of that integer (e.g. 12 could mean the location (4,14))
        value_names = {
            variables[0]: list(locations),
            variables[1]: [name for name in containers]
        }
        for i, loc_i in enumerate(value_names[variables[0]]):
            for j, cont_j in enumerate(value_names[variables[1]]):
                is_in = self.is_in(loc_i, containers[cont_j])
                if not self.negate:
                    potential = 1.0-1e-9 if is_in else 1e-9
                else:
                    # negation
                    potential = 1e-9 if is_in else 1.0-1e-9
                potentials.append(potential)                    
        factor = DiscreteFactor(variables, cardinality=[item_card, container_card],
                                values=potentials, state_names=value_names)
        return factor                
                 
        

    # def to_mrf(self):
    #     factor = self.to_factor()
    #     G = MarkovModel()
    #     G.add_nodes_from(variables)
    #     G.add_edges_from(edges)
    #     G.add_factors(factor)
    #     assert G.check_model()
    #     return SemanticMRF(G, value_names)
        
