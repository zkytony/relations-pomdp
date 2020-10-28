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
from relpomdp.home2d.domain.action import *
from relpomdp.home2d.domain.state import *
# from relpomdp.home2d.reward_model import *
from relpomdp.home2d.domain.condition_effect import *
from relpomdp.home2d.domain.relation import *
from relpomdp.home2d.domain.maps.grid_map import *
from relpomdp.home2d.domain.visual import Home2DViz
from relpomdp.oopomdp.framework import *
import pomdp_py

class Home2DEnvironment(OOEnvironment):
    """Single robot home environment"""

    def __init__(self,
                 robot_id,
                 grid_map,  # specifies the map (dimension and walls)
                 init_object_states, # maps from object id to initial state
                 motions={MoveW, MoveN, MoveS, MoveE},
                 reward_model=CompositeRewardModel([])):
        """
        reward_model (RewardModel): May differ depending on the task.
            default is a dummy reward model that always returns 0.
        """
        self.robot_id = robot_id
        self.grid_map = grid_map
        self.class_to_ids = {}
        for objid in init_object_states:
            objclass = init_object_states[objid].objclass
            if objclass not in self.class_to_ids:
                self.class_to_ids[objclass] = set()
            self.class_to_ids[objclass].add(objid)

        init_state = OOState(init_object_states)
        relations = {touch_N,
                     touch_S,
                     touch_W,
                     touch_E,
                     is_on}
        self.legal_motions = grid_map.compute_legal_motions(motions)
        # Basic condition/effects
        cond_effects = [(CanMove(robot_id, self.legal_motions), MoveEffect(robot_id))]
        super().__init__(init_state, relations, cond_effects, reward_model)

    def ids_for(self, objclass):
        return self.class_to_ids.get(objclass, set())

    @property
    def robot_state(self):
        return self.state.object_states[self.robot_id]

    @property
    def width(self):
        return self.grid_map.width

    @property
    def length(self):
        return self.grid_map.length

    def add_reward_model(self, reward_model):
        self.reward_model.add_model(reward_model)


def unittest():
    from relpomdp.home2d.domain.maps import all_maps
    grid_map = all_maps["map_small_0"]()

    # Building object state
    robot_id = 1
    robot_state = Objstate("Robot",
                           pose=(0,0,0),
                           camera_direction="+x")
    salt_id = 10
    salt_state = Objstate("Salt",
                          pose=(3,3))

    pepper_id = 15
    pepper_state = Objstate("Pepper",
                            pose=(3,2))

    init_state = {robot_id: robot_state,
                  salt_id: salt_state,
                  pepper_id: pepper_state}
    env = Home2DEnvironment(robot_id,
        grid_map, init_state)

    viz = Home2DViz(env,
                    {salt_id: (128, 128, 128),
                     pepper_id: (200, 10, 10)},
                    res=40,
                    controllable=True)
    viz.on_init()
    viz.on_execute()


if __name__ == "__main__":
    unittest()
