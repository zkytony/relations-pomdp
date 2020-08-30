from relpomdp.object_search.action import *
from relpomdp.object_search.state import *
from relpomdp.object_search.reward_model import *
from relpomdp.object_search.policy_model import *
from relpomdp.object_search.condition_effect import *
from relpomdp.object_search.relation import *
from relpomdp.object_search.grid_map import *
from relpomdp.object_search.visual import ObjectSearchViz
from relpomdp.oopomdp.framework import *
import pomdp_py

class ObjectSearchAgent(OOAgent):
    def __init__(self, grid_map, sensor, ids,
                 init_belief):
        
        mp = MotionPolicy(grid_map)        
        cond_effects_t = {(CanMove(ids, mp), MoveEffect(ids)),
                          (CanPickup(ids), PickupEffect())}
        cond_effects_o = {(CanObserve(ids), ObserveEffect(sensor, ids, epsilon=1e-12))}
        action_prior = GreedyActionPrior(ids, mp, 10, 100)
        policy_model = PreferredPolicyModel(action_prior)
        # policy_model = PolicyModel(ids, mp)
        reward_model = RewardModel(ids)
        self.ids = ids
        super().__init__(init_belief, cond_effects_t, cond_effects_o,
                         policy_model, reward_model)
