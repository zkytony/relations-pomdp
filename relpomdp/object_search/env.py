# Environment for 2D object search. This is different from
# the MOS in pomdp_py in the following aspects:
#
# - Instead of a grid world with obstacles as grid cells,
#   This grid world contains walls and is more natural to
#   model an office or home environment.
#
# - This environment is built on top of the relation/condition/effect
#   language of OO-MDP. So the way transition dynamics is specified
#   is quite different -- And no object independence assumption is
#   used while building this domain!
#
# - There is an emphasis on domain scale as well as a diverse set
#   of object classes. This is to facilitate abstraction/hierarchical
#   POMDP planning.
#
from relpomdp.object_search.action import *
from relpomdp.object_search.state import *
from relpomdp.object_search.reward_model import *
from relpomdp.object_search.condition_effect import *
from relpomdp.object_search.relation import *
from relpomdp.object_search.grid_map import *
from relpomdp.object_search.visual import ObjectSearchViz
from relpomdp.oopomdp.framework import *
import pomdp_py

class ObjectSearchEnvironment(OOEnvironment):
    
    def __init__(self,
                 grid_map,  # specifies the map (dimension and walls)
                 init_object_states, # maps from object id to initial state
                 target_objects):  # a set of object ids that are the targets
        
        # maps from object class to id        
        ids = {}  
        for objid in init_object_states:
            c = init_object_states[objid].objclass
            if c not in ids:
                ids[c] = []
            ids[c].append(objid)
        ids["Robot"] = ids["Robot"][0]
        ids["Target"] = target_objects
        self.ids = ids
        self.grid_map = grid_map

        init_state = JointState(init_object_states)
        relations = {touch_N,
                     touch_S,
                     touch_W,
                     touch_E,
                     is_on}
        mp = MotionPolicy(grid_map)
        cond_effects = {(CanMove(ids, mp), MoveEffect(ids)),
                        (CanPickup(ids), PickupEffect())}
        reward_model = RewardModel(ids)
        super().__init__(init_state, relations, cond_effects, reward_model)

    @property
    def robot_state(self):
        return self.state.object_states[self.ids["Robot"]]

    @property
    def width(self):
        return self.grid_map.width

    @property
    def length(self):
        return self.grid_map.length 


if __name__ == "__main__":
    # Building grid map
    top_walls = [(x,4,"H") for x in range(5)]
    bottom_walls = [(x,-1,"H") for x in range(5)]
    left_walls = [(-1,y,"V") for y in range(5)]
    right_walls = [(4,y,"V") for y in range(5)]
    inner_walls = [(0,0,"V"), (0,1,"V"), (1,3,"V"), (1,4,"V"), (2,0,"V"), (2,1,"V")]
    walls = top_walls + bottom_walls + left_walls + right_walls\
        + inner_walls
    
    wall_states = {}
    for i, tup in enumerate(walls):
        x, y, direction = walls[i]
        wall_states[1000+i] = WallState((x,y), direction)
        
    grid_map = GridMap(5, 5, wall_states, [])

    # Building object state
    robot_state = RobotState((0,0,0), "+x")
    salt_state = ItemState("Salt", (3,3))
    pepper_state = ItemState("Pepper", (3,2))
    init_state = {1: robot_state,
                  10: salt_state,
                  15: pepper_state}
    env = ObjectSearchEnvironment(grid_map,
                                  init_state,
                                  {10})
    viz = ObjectSearchViz(env,
                          {10: (128, 128, 128),
                           15: (200, 10, 10)},
                          res=40,
                          controllable=True)
    viz.on_init()
    viz.on_execute()
