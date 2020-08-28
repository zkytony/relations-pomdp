import relpomdp.oopomdp.framework as oopomdp
from relpomdp.object_search.state import WallState
from relpomdp.object_search.utils import euclidean_dist
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.models import MarkovModel
from pgmpy.inference import BeliefPropagation

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
class Near(oopomdp.InfoRelation):
    def __init__(self, class1, class2, grid_map):
        self.grid_map = grid_map
        super().__init__("near",
                         class1, class2)
        
    def eval(self, object_state1, object_state2):
        """Returns True if the Relation holds. False otherwise.
        According to the paper, on(o1,o2) holds if o1 and o2 are
        overlapping"""
        if object_state1.objclass == self.class1.name\
           and object_state2.objclass == self.class2.name:
            return self.is_near(object_state1.pose, object_state2.pose)
        return False

    def is_near(self, pose1, pose2):
        same_room = (self.grid_map.room_of(pose1)\
                         == self.grid_map.room_of(pose2))
        return same_room\
            and euclidean_dist(pose1, pose2) <= 2

    def to_mrf(self):
        locations = [(x,y)
                     for x in range(self.grid_map.width)\
                     for y in range(self.grid_map.length)]
        card = len(locations)
        variables = ["%s_Pose" % self.class1.name,
                     "%s_Pose" % self.class2.name]
        edges = [[variables[0], variables[1]]]

        # potentials: a list of joint potentials for the table.
        potentials = []
        
        # value_names: pgmpy's discrete factor expects every value be
        # indexed by an integer. Here we are recording the actual meaning
        # of that integer (e.g. 12 could mean the location (4,14))
        value_names = {
            variables[i]: list(locations)
            for i in range(len(variables))
        }
        
        # semantics = {}  # tabular entry (i,j) to semantic meaning (val_i, val_j)
        # index_to_semantics = []
        for i, loc_i in enumerate(locations):
            for j, loc_j in enumerate(locations):
                # semantics[(i,j)] = (loc_i, loc_j)
                near = self.is_near(loc_i, loc_j)
                if near:
                    potentials.append(1.0-1e-9)
                else:
                    potentials.append(1e-9)
                    
        factor = DiscreteFactor(variables, cardinality=[card, card],
                                values=potentials, state_names=value_names)
        G = MarkovModel()
        G.add_nodes_from(variables)
        G.add_edges_from(edges)
        G.add_factors(factor)
        assert G.check_model()
        return SemanticMRF(G, value_names)

class SemanticMRF:
    def __init__(self, markov_model, value_to_name):
        """semantics_semantics (dict): map from tuple of integer indices (indicating variable values)
                to a meaningful value of the variable. Note that both the index and
                the value must be unique"""
        self.markov_model = markov_model

        self.value_to_name = value_to_name  # {variable -> {value_index -> value_name}}
        self.name_to_value = {}  # {variable -> {value_name -> value_index}}
        for variable in self.value_to_name:
            self.name_to_value[variable] =\
                {self.value_to_name[variable][value_index]:value_index
                 for value_index in range(len(self.value_to_name[variable]))}
        self.bp = BeliefPropagation(self.markov_model)

    @property
    def G(self):
        return self.markov_model

    @property
    def factors(self):
        return self.G.factors

    def query(self, variables, evidence=None):
        """
        evidence is a mapping from variable to value_name. The value_name
            is the semantic one - e.g. for location, it's (x,y). Its
            integer index value in the MRF model will be used for
            actual inference.
        """
        for variable in variables:
            if not self.valid_var(variable):
                raise ValueError("Variable %s is not in the model" % variable)
        phi = self.bp.query(variables, evidence=evidence)
        return phi

    def valid_var(self, var):
        return var in set(self.G.nodes)

    def values(self, var):
        return list(self.name_to_value[var].keys())
    
