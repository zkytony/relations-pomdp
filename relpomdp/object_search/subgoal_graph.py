# Methods to construct a subgoal graph
from relpomdp.oopomdp.graph import Graph
from relpomdp.oopomdp.framework import Class
from relpomdp.object_search.subgoals import *
from relpomdp.object_search.info_relations import *
from relpomdp.utils import perplexity

class SubgoalGraph(Graph):
    def __init__(self, relations):
        super().__init__(relations)
        self.subgoal_class_nodes = []
        for nid in self.nodes:
            if isinstance(self.nodes[nid], SubgoalClass):
                self.subgoal_class_nodes.append(nid)

    def ground(self, class_to_args, relation_to_args):
        """
        Grounds this graph
        Args:
            class_to_args (dict): Maps from string (corresponding to a SubgoalClass)
                to a tuple (args, kwargs) needed to ground this subgoal class.

            relation_to_args (dict): Maps from a tuple (relation id) to a tuple (args, kwargs)
                to ground it. Note that the relation has 'id' attribute which will
                be used to refer to the edge inside this graph
        """
        for c in class_to_args:
            if c in self.nodes:
                args, kwargs = class_to_args[c]
                self.nodes[c].ground(*args, **kwargs)
        for r in relation_to_args:
            if r in self.edges:
                args, kwargs = relation_to_args[r]
                self.edges[r].ground(*args, **kwargs)

    def suggest_subgoal(self, target_class, robot_belief):
        """
        Args:
            robot_belief (OOBelief)
        """
        candidates = []
        for nid in self.subgoal_class_nodes:
            subgoal_class = self.nodes[nid]
            for subgoal_name in subgoal_class.subgoals:
                subgoal = subgoal_class.subgoals[subgoal_name]
                if subgoal.valid(robot_belief):
                    candidates.append(subgoal)

        best_subgoal = None
        lowest_plx = float("inf")
        for subgoal in candidates:
            # Values maps from (obs_class, observation) -> probability
            final_dist = {}
            values = subgoal.effect(robot_belief)
            for obs_class, observation in values:
                prob = values[(obs_class, observation)]
                dist = self.query(target_class, {obs_class:observation})
                for target_observation in dist:
                    final_dist[target_observation] =\
                        final_dist.get(target_observation, 1.0)\
                        * prob * dist[target_observation]
            total_prob = 0.0
            for o in final_dist:
                total_prob += final_dist[o]
            probs = []
            for o in final_dist:
                final_dist[o] /= total_prob
                probs.append(final_dist[o])
            plx = perplexity(probs)
            if plx < lowest_plx:
                best_subgoal = subgoal
                lowest_plx = plx
        return best_subgoal

    def query(self, class_name, evidence):
        """
        Returns a distribution over objects under the class (i.e.
        different class attribute settings) based on object 
        relations, given evidence.
        
        Args:
            class_name (str): Class whose distribution is of interest
            evidence (dict): dictionary mapping from class name to observation
        """
        # TODO: This should be done by belief propagation
        # Find the relations connecting evidence and the class
        # Compute probability of every object under class
        # Normalize
        class_node = self.nodes[class_name]
        dist = {}
        for ev_class in evidence:
            ev_observation = evidence[ev_class]
            ev_class_node = self.nodes[ev_class]
            edges = self.edges_between(class_node.id, ev_class_node.id)
            if edges is None:
                continue
            assert len(edges) == 1, "RIGHT NOW ONLY CONSIDER ONE EDGE BETWEEN TWO OBJECTS"
            eid = list(edges)[0]
            relation = self.edges[eid]
            for class_observation in class_node.observation_variations():
                if class_name == ev_class:
                    # You actually observed this class
                    if ev_observation == class_observation:
                        dist[class_observation] = 1.0 - 1e-9
                    else:
                        dist[class_observation] = 1e-9
                else:
                    prob = relation.probability(class_observation, ev_observation)
                    if class_observation not in dist:
                        dist[class_observation] = 1.0
                    dist[class_observation] *= prob
        return dist
        

# Test
def unittest():
    from relpomdp.object_search.tests.salt_pepper import office_floor1 as salt_pepper_1
    from relpomdp.object_search.env import ObjectSearchEnvironment    
    kitchen = RoomClass("Kitchen")
    office = RoomClass("Office")
    salt = ItemClass("Salt")
    pepper = ItemClass("Pepper")
    computer = ItemClass("Computer")

    salt_in_kitchen         = RoomLevelIn(salt, kitchen)
    salt_not_in_office      = RoomLevelIn(salt, office, negate=True)
    pepper_in_kitchen       = RoomLevelIn(pepper, kitchen)
    pepper_not_in_office    = RoomLevelIn(pepper, office, negate=True)
    computer_not_in_kitchen = RoomLevelIn(computer, kitchen, negate=True)
    computer_in_office      = RoomLevelIn(computer, office)
    salt_not_near_computer  = GroundLevelNear(salt, computer, negate=True)
    salt_near_pepper        = GroundLevelNear(salt, pepper)

    relations = {salt_in_kitchen         ,
                 salt_not_in_office      ,
                 pepper_in_kitchen       ,
                 pepper_not_in_office    ,
                 computer_not_in_kitchen ,
                 computer_in_office      ,
                 salt_not_near_computer  ,
                 salt_near_pepper        }
    graph = SubgoalGraph(relations)
    print("---- Edges ----")
    print(graph.edges)
    print("---- Nodes ----")    
    print(graph.nodes)
    for nid in graph.nodes:
        print(nid, graph.nodes[nid].grounded)
    for eid in graph.edges:
        print(eid, graph.edges[eid].grounded)                

    print("** Grounding **")
    ids, grid_map, init_state, colors = salt_pepper_1()
    cargs = {
        "Kitchen": ([grid_map, ids], {"knows_room_types": False}),
        "Office": ([grid_map, ids], {"knows_room_types": False}),
        "Salt": ([grid_map], {}),
        "Pepper": ([grid_map], {}),
        "Computer": ([grid_map], {}),        
    }
    rargs = {}
    for r in relations:
        rargs[r.id] = [[grid_map], {}]
    graph.ground(cargs, rargs)
    for nid in graph.nodes:
        print(nid, graph.nodes[nid].grounded)
    for eid in graph.edges:
        print(eid, graph.edges[eid].grounded)        

    # Construct environment
    env = ObjectSearchEnvironment(ids,
                                  grid_map,
                                  init_state)            

    # Obtain prior
    target_hist = {}
    total_prob = 0
    prior_type = "uniform"
    target_class = "Salt"
    target_id = ids["Target"][0]
    robot_id = ids["Robot"]
    for x in range(grid_map.width):
        for y in range(grid_map.length):
            state = ItemState(target_class, (x,y))
            if prior_type == "uniform":
                target_hist[state] = 1.0
            elif prior_type == "informed":
                if (x,y) != env.state.object_states[target_id].pose.value:
                    target_hist[state] = 0.0
                else:
                    target_hist[state] = 1.0
            total_prob += target_hist[state]
    # Normalize
    for state in target_hist:
        target_hist[state] /= total_prob

    init_belief = pomdp_py.OOBelief({target_id: pomdp_py.Histogram(target_hist),
                                     robot_id: pomdp_py.Histogram({env.robot_state:1.0})})

    print(graph.suggest_subgoal(target_class, init_belief))
            
    
if __name__ == "__main__":
    unittest()
