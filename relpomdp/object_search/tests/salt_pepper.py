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


def office_floor1(init_robot_pose=(9,0,0)):
    """
    Office floor with salt, pepper, and computers
    This describes the full environment configuration (i.e. all
    object locations and the robot's location).
    """
    grid_map = small_map1()  # THiS IS FIXED - office_floor1 uses small_map1
    init_salt_pose = Pose((0,6))
    init_pepper_pose = Pose((1,5))
    init_robot_pose = Pose(init_robot_pose)
    start_room = grid_map.rooms[grid_map.room_of(init_robot_pose[:2])]

    salt_id = 10
    pepper_id = 15
    robot_id = 1    
    robot_state = RobotState(init_robot_pose, "-x",
                             start_room.room_type)
    salt_state = ItemState("Salt", init_salt_pose)
    pepper_state = ItemState("Pepper", init_pepper_pose)
    
    computer_poses = [(5,9), (6,2)]
    computer_states = []
    for pose in computer_poses:
        computer_states.append(ItemState("Computer", Pose(pose)))
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
    colors = {"Salt": (128, 128, 128),
              "Pepper": (200, 10, 10)}

    return ids, grid_map, init_state, colors
