# Use a relational graph actively to generate subgoals for grounding.
from relpomdp.home2d.planning.relations import *
from relpomdp.oopomdp.infograph import *
from relpomdp.pgm.mrf import SemanticMRF, factors_to_mrf
from relpomdp.utils import perplexity
from relpomdp.home2d.tasks.search_item.search_item_task import SearchItemTask
from relpomdp.home2d.tasks.search_room.search_room_task import SearchRoomTask

def next_grounding_task(belief,
                        b_attr,  # attribute in the belief that wants to be grounded
                        rel_graph,
                        grid_map,
                        current_tasks=set()):
    # Get neighbors
    scores = {}  # Choose one with lowest score
    for eid in rel_graph.edges_from(b_attr.id):
        edge = rel_graph.edges[eid]
        other_attr = edge.other_node(b_attr.id)
        if edge.potential is None:
            edge.ground_on_map(grid_map)  # ground this edge
            potential = edge.grounding_to_potential()        

        # Get MRF, 
        print("Building MRF")
        mrf = factors_to_mrf([edge.potential])
        print("Inference")
        phi = mrf.query([b_attr.id])

        values = []
        for s in belief:
            values.append(phi.get_value({b_attr.id:s[b_attr.name]}) * belief[s])
        
        plx = perplexity(values)

        # if isinstance(other_attr, PoseAttr):
        #     difficulty = 100
        if isinstance(other_attr, RoomAttr):
        #     difficulty = 50
            scores[eid] = plx

    chosen_eid = min(scores, key=lambda eid:scores[eid])
    edge = rel_graph.edges[chosen_eid]
    other_attr = edge.other_node(b_attr.id)    
    # if isinstance(other_attr, PoseAttr):
    #     task = SearchItemTask
    if isinstance(other_attr, RoomAttr):
        task = SearchRoomTask(other_attr.clas)
    return task
