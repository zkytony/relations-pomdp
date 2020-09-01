from relpomdp.object_search.world_specs.build_world import *
from relpomdp.object_search.env import *
from relpomdp.object_search.sensor import *
from relpomdp.object_search.agent import *
from relpomdp.object_search.greedy_planner import GreedyPlanner
from relpomdp.pgm.mrf import SemanticMRF, relations_to_mrf
from relpomdp.object_search.relation import *
import networkx as nx
import matplotlib.pyplot as plt
from pgmpy.inference import BeliefPropagation
from search_and_rescue.experiments.plotting import *
import pomdp_py
import pygame
import time
import pickle
import os


def office_floor1(init_robot_pose=(9,0,0),
                  mrfdir="./mrf", save_mrf=True):
    """
    Office floor with salt, pepper, and computers
    This describes the full environment configuration (i.e. all
    object locations and the robot's location).
    """
    grid_map = small_map1()  # THiS IS FIXED - office_floor1 uses small_map1
    init_salt_pose = (0,6)
    init_pepper_pose = (1,5)

    salt_id = 10
    pepper_id = 15
    robot_id = 1    
    robot_state = RobotState(init_robot_pose, "-x")
    salt_state = ItemState("Salt", init_salt_pose)
    pepper_state = ItemState("Pepper", init_pepper_pose)
    
    computer_poses = [(5,9), (6,2)]
    computer_states = []
    for pose in computer_poses:
        computer_states.append(ItemState("Computer", pose))
    init_state = {robot_id: robot_state,
                  pepper_id: pepper_state,
                  salt_id: salt_state}
    for i, s in enumerate(computer_states):
        init_state[3000+i] = s

    # maps from object class to id        
    ids = {}  
    for objid in init_state:
        c = init_state[objid].objclass
        if c not in ids:
            ids[c] = []
        ids[c].append(objid)
    ids["Robot"] = ids["Robot"][0]
    ids["Target"] = [salt_id]

    # Relations in this world;
    # Check if the MRF already exists
    # mrf_path = os.path.join(mrfdir, "salt_pepper_1.pkl")
    # if os.path.exists(mrf_path):
    #     with open(mrf_path, "rb") as f:
    #         mrf = pickle.load(f)
    # else:
    near_salt_pepper = Near("Salt", "Pepper", grid_map)  # grounding the relation on the grid map
    not_near_salt_computer = Near("Salt", "Computer", grid_map, negate=True)
    in_salt_kitchen = In("Salt", "Kitchen", "Room", grid_map)
    mrf = relations_to_mrf([near_salt_pepper])#, in_salt_kitchen])
        # if save_mrf:
        #     os.makedirs(mrfdir, exist_ok=True)
        #     with open(mrf_path, "wb") as f:
        #         pickle.dump(mrf, f)
    
    colors = {"Salt": (128, 128, 128),
              "Pepper": (200, 10, 10)}
    return ids, grid_map, init_state, mrf, colors
