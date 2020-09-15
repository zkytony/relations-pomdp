# Methods to construct a subgoal graph
from relpomdp.oopomdp.graph import Graph
from relpomdp.oopomdp.framework import Class
from relpomdp.object_search.subgoals import *
from relpomdp.object_search.info_relations import *

class SubgoalGraph(Graph):
    def __init__(self, relations):
        super().__init__(relations)
        self.subgoal_class_nodes = []
        for nid in self.nodes:
            if isinstance(self.nodes[nid], SubgoalClass):
                self.subgoal_class_nodes.append(nid)

    def ground(self, class_to_args):
        """
        Grounds this graph
        Args:
            class_to_args (dict): Maps from string (corresponding to a SubgoalClass)
                to a tuple (args, kwargs) needed to ground this subgoal class.
        """
        for c in class_to_args:
            if isinstance(self.nodes[c], SubgoalClass):
                args, kwargs = class_to_args[c]
                self.nodes[c].ground(*args, **kwargs)

# Test
def unittest():
    from relpomdp.object_search.world_specs.build_world import small_map1
    kitchen = RoomClass("Kitchen")
    office = RoomClass("Office")
    salt = Class("Salt")
    pepper = Class("Pepper")
    computer = Class("Computer")

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
        if isinstance(graph.nodes[nid], SubgoalClass):
            print(nid, graph.nodes[nid].grounded)

    print("** Grounding **")
    ids = {"Pepper": [15],
           "Salt": [10],
           "Robot": 5,
           "Target": [10]}
    grid_map = small_map1()
    cargs = {
        "Kitchen": ([grid_map, ids], {"knows_room_types": False}),
        "Office": ([grid_map, ids], {"knows_room_types": False})
    }
    graph.ground(cargs)
    for nid in graph.nodes:
        if isinstance(graph.nodes[nid], SubgoalClass):
            print(nid, graph.nodes[nid].grounded)    
    
if __name__ == "__main__":
    unittest()
