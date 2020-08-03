from search_and_rescue import *
from search_and_rescue.experiments.trial import *
from dynamic_mos.experiments.world_types import create_free_world
from search_and_rescue.utils import place_objects
import numpy as np
import pgmpy
import random
from pgmpy.models import MarkovModel
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.inference import BeliefPropagation

map_dim = (10, 10)
card = map_dim[0]*map_dim[1]
objects = ["A", "B", "C", "D"]
edges = [["A", "B"],
         ["B", "C"],
         ["B", "D"],
         ["C", "D"]]
factor_ab = DiscreteFactor(['A','B'], cardinality=[card, card], values=np.random.rand(card**2))
factor_bc = DiscreteFactor(['B','C'], cardinality=[card, card], values=np.random.rand(card**2))
factor_cd = DiscreteFactor(['C','D'], cardinality=[card, card], values=np.random.rand(card**2))
factor_bd = DiscreteFactor(['B','D'], cardinality=[card, card], values=np.random.rand(card**2))

G = MarkovModel()
G.add_nodes_from(objects)
G.add_edges_from(edges)
G.add_factors(factor_ab, factor_bc, factor_cd, factor_bd)
G.check_model()

random.seed(100)
# Create world
mapstr, free_locations = create_free_world(*map_dim) # create_hallway_world(9, 2, 1, 3, 3)
# mapstr, free_locations = create_free_world(10,10)
# mapstr, free_locations = create_free_world(10,10)#create_connected_hallway_world(9, 1, 1, 3, 3)#create_free_world(6, 6)
#create_connected_hallway_world(9, 1, 1, 3, 3) # #create_two_room_loop_world(5,5,3,1,1)#create_two_room_world(4,4,3,1) #create_free_world(6,6)#
object_poses = []
for obj in objects:
    pose = random.sample(free_locations, 1)[0]
    object_poses.append((obj, pose))
    free_locations = free_locations - set({pose})
robot_pose = random.sample(free_locations, 1)[0]
free_locations = free_locations - set({robot_pose})
object_poses.append(("R", robot_pose))

laserstr = make_laser_sensor(90, (1, 2), 0.5, False)
mapstr = place_objects(mapstr, object_poses)
worldstr = equip_sensors(mapstr, {"R": laserstr})
problem_args = {"can_stay": False,
                "mdp_agent_ids": {},
                "look_after_move": True}
solver_args = {"visualize": True,
               "planning_time": 0.7,
               "exploration_const": 200,
               "discount_factor": 0.95,
               "max_depth": 10,
               "greedy_searcher": False,
               "game_mode": True,
            "controller_id": None}
config = {"problem_args": problem_args,
          "solver_args": solver_args,
        "world": worldstr}

trial = SARTrial("trial_0_test", config, verbose=True)
try:
    trial.run(logging=True)
except KeyboardInterrupt:
    print("Stopping...")
finally:
    trial.objects["viz"].replay(interval=0.3)
        

