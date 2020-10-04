from relpomdp.home2d.action import *
from relpomdp.home2d.state import *
from relpomdp.home2d.reward_model import *
from relpomdp.home2d.policy_model import *
from relpomdp.home2d.condition_effect import *
from relpomdp.home2d.relation import *
from relpomdp.home2d.grid_map import *
from relpomdp.home2d.visual import ObjectSearchViz
from relpomdp.oopomdp.framework import *
import pomdp_py

class ObjectSearchAgent(OOAgent):
    def __init__(self, grid_map, sensor, ids,
                 init_belief, mrf):
        
        mp = MotionPolicy(grid_map)
        cond_effects_t = [(CanMove(ids, mp), MoveEffect(ids)),
                          (CanPickup(ids), PickupEffect())]
        cond_effects_o = [(CanObserve(ids), ObjectObserveEffect(sensor, ids, epsilon=1e-12)),
                          (CanObserve(ids), RoomObserveEffect(sensor, ids, grid_map, epsilon=1e-12))]
        action_prior = GreedyActionPrior(ids, mp, 10, 100)
        policy_model = PreferredPolicyModel(action_prior)
        # policy_model = PolicyModel(ids, mp)
        reward_model = RewardModel(ids)
        self.ids = ids
        self.sensor = sensor
        self.grid_map = grid_map
        self.mrf = mrf
        super().__init__(init_belief, cond_effects_t, cond_effects_o,
                         policy_model, reward_model)
    
