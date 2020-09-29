# Use a relational graph actively to generate subgoals for grounding.
from relpomdp.home2d.planning.relations import *
from relpomdp.oopomdp.infograph import *
from relpomdp.pgm.mrf import SemanticMRF, factors_to_mrf
from relpomdp.utils import perplexity
from relpomdp.home2d.tasks.search_item.search_item_task import SearchItemTask
from relpomdp.home2d.tasks.search_room.search_room_task import SearchRoomTask
import random


def get_domain(attr, grounding):
    d1, d2, _ = grounding
    if d1.attr == attr:
        return d1
    elif d2.attr == attr:
        return d2
    else:
        raise ValueError("Attribute %s's domain is not in grounding." % (attr.id))

# TODOS:
# - Right now this can only ground the task of searching for a room.
# - Right now the task choice is only based on perplexity (which is
#   easy enough to be computed for a room)
def next_grounding_task(belief,
                        b_attr,  # attribute in the belief that wants to be grounded
                        rel_graph,
                        grid_map,
                        robot_id,
                        current_tasks=set()):
    # Get neighbors
    perps = {}  # Choose one with lowest score
    for eid in rel_graph.edges_from(b_attr.id):
        edge = rel_graph.edges[eid]
        other_attr = rel_graph.nodes[edge.other_node(b_attr.id)]
        if edge.potential is None:
            edge.ground_on_map(grid_map)  # ground this edge
            potential = edge.grounding_to_potential()        

        # Get MRF, 
        print("Building MRF")
        mrf = factors_to_mrf([edge.potential])

        # Perform inference. Hypothesize a value for the other attribute
        print("Inference")
        dom = get_domain(other_attr, edge.grounding)
        val = random.sample(dom.values(), 1)[0]
        phi = mrf.query([b_attr.id], evidence={other_attr.id:val})

        values = []
        tot_prob = 0
        for s in belief:
            values.append(phi.get_value({b_attr.id:s[b_attr.name]}) * belief[s])
            tot_prob += values[-1]
            
        for i in range(len(values)):
            values[i] /= tot_prob
            
        plx = perplexity(values)

        # if isinstance(other_attr, PoseAttr):
        #     difficulty = 100
        if isinstance(other_attr, RoomAttr):
        #     difficulty = 50
            perps[eid] = plx

    chosen_eid = min(perps, key=lambda eid:perps[eid])
    edge = rel_graph.edges[chosen_eid]
    other_attr = rel_graph.nodes[edge.other_node(b_attr.id)]
    # if isinstance(other_attr, PoseAttr):
    #     task = SearchItemTask
    if isinstance(other_attr, RoomAttr):
        task = SearchRoomTask(robot_id, other_attr.clas, grid_map=grid_map)
    return task, perps[chosen_eid]
