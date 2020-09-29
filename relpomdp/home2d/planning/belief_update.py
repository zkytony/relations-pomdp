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
from relpomdp.pgm.mrf import SemanticMRF
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.models import MarkovModel
from pgmpy.inference import BeliefPropagation
from relpomdp.home2d.planning.relations import *
from relpomdp.oopomdp.infograph import *
from relpomdp.home2d.utils import objstate, objobs, ooobs, euclidean_dist

def belief_update(belief,
                  b_attr,
                  relgraph,
                  observation,
                  o_attr,
                  grid_map):
    
    # TODO: Implement the case for indirect connection
    import pdb; pdb.set_trace()
    edges = relgraph.edges_between(b_attr.id, o_attr.id)
    if edges is None or len(edges) > 1:
        raise ValueError("This case is not implemented yet.")

    edge = relgraph.edges[list(edges)[0]]  # TODO: change
    edge.ground_on_map(grid_map)  # ground this edge

    potential = edge.grounding_to_potential()
    # Build MRF potential


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
    belief_update(None, salt, graph, objobs("Pepper", pose=(0,1)), pepper, grid_map)

if __name__ == "__main__":
    unittest()    
