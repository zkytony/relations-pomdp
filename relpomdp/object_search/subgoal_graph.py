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



# Test
def unittest():
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

if __name__ == "__main__":
    unittest()
