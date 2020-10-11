# Uses a relational graph passively for belief update
#
# The passive belief updater works as follows:
#
# 1. A relational graph over attributes
# 2. The belief over one attribute value is given
# 3. The observation of one attribute in the graph is given
# 4. The observed attributes are grounded
# 5. Paths from each observed attribute to the belief variable is computed
# 6. Marginal inference is performed for the modeled variable.
# 7. This marginal probability is multiplied with the current belief
#    to produce the new probability distribution
#
# Relevant paper: https://www.ijcai.org/Proceedings/2018/0692.pdf
from relpomdp.pgm.mrf import SemanticMRF, factors_to_mrf
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.models import MarkovModel
from pgmpy.inference import BeliefPropagation
from relpomdp.home2d.planning.iter0.relations import *
from relpomdp.oopomdp.infograph import *
from relpomdp.home2d.utils import euclidean_dist
from relpomdp.oopomdp.framework import Objstate, Objobs, OOObs
import pomdp_py

# TODO:
# - Right now only direct relation is considered
# - Need a mechnism for the situation where a subtask evidence
#   no longer contributes to solving the global task --- This
#   could be done by giving the evidence to the next_grounding_subtask
#   function, and also specifying that obtaining the same evidence
#   does not solve the task, when creating the subtask in next_grounding_subtask.
#   This would then require modification to the SearchRoomTask class (for example).
#   to account for the already observed evidence.
# - Right now the evidence_from_subtask only allows one evidence per subtask.
#   If say, we want to use the evidence of two office rooms, then we need to
#   operate on the grounded relational graph where there are two office room nodes...
#   This seems complicated. Maybe the old evidence have already contributed
#   to the belief update, so we don't need to use them again, and we only need
#   to track the most recent evidence for an attribute.
def relation_belief_update(belief,
                           b_attr,
                           relgraph,
                           observation,
                           o_attr,
                           grid_map):
    
    # TODO: Implement the case for indirect connection
    edges = relgraph.edges_between(b_attr.id, o_attr.id)
    if edges is None or len(edges) > 1:
        print("WARNING: This case is not implemented yet.")
        return belief

    edge = relgraph.edges[list(edges)[0]]  # TODO: change
    if edge.potential is None:
        edge.ground_on_map(grid_map)  # ground this edge
        potential = edge.grounding_to_potential()
    # Build MRF potential
    print("Building MRF")
    mrf = factors_to_mrf([edge.potential])
    print("Inference")
    phi = mrf.query([b_attr.id], evidence={o_attr.id:observation[o_attr.name]})

    new_belief = {}
    for s in belief:
        new_belief[s] = phi.get_value({b_attr.id:s[b_attr.name]}) * belief[s]
    return pomdp_py.Histogram(new_belief)
    
#### Unit test
def unittest():
    from relpomdp.home2d.domain.maps import all_maps    
    pepper = PoseAttr("Pepper")
    salt = PoseAttr("Salt")
    kitchen = PoseAttr("Kitchen")
    near1 = Near(pepper, salt)
    near2 = Near(salt, pepper)
    in1 = In(pepper, kitchen)
    in2 = In(salt, kitchen)
    graph = RelationGraph({in1, in2, near1, near2})
    grid_map = all_maps["map_small_1"]()    
    relation_belief_update(None, salt, graph, Objobs("Pepper", pose=(0,1)), pepper, grid_map)

if __name__ == "__main__":
    unittest()    