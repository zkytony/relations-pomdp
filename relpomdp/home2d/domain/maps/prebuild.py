from relpomdp.home2d.domain.maps import all_maps
from relpomdp.oopomdp.infograph import *
from relpomdp.pgm.mrf import SemanticMRF, factors_to_mrf
from relpomdp.utils import perplexity
from relpomdp.home2d.tasks.search_item.search_item_task import SearchItemTask
from relpomdp.home2d.tasks.search_room.search_room_task import SearchRoomTask
from relpomdp.home2d.utils import euclidean_dist, save_images_and_compress
from relpomdp.oopomdp.framework import Objstate, Objobs, OOObs
from relpomdp.home2d.tasks.common.sensor import *
from relpomdp.home2d.domain.env import Home2DEnvironment
from relpomdp.home2d.tasks.search_item.search_item_task import *
from relpomdp.home2d.tasks.search_item.visual import SearchItemViz
from relpomdp.utils import perplexity
import subprocess
import pomdp_py

from relpomdp.home2d.domain.maps.grid_map import GridMap
from relpomdp.home2d.domain.maps.build_map import *

def office_floor1_1(init_robot_pose=(9,0,0)):
    """
    Office floor with salt, pepper, and computers
    This describes the full environment configuration (i.e. all
    object locations and the robot's location).
    """
    grid_map = all_maps["map_small_1"]()    
    init_salt_pose = (0,6)
    init_pepper_pose = (1,5)
    init_robot_pose = init_robot_pose

    salt_id = 10
    pepper_id = 15
    robot_id = 1
    robot_state = Objstate("Robot",
                           pose=init_robot_pose,
                           camera_direction="-x")
    salt_state = Objstate("Salt",
                          pose=init_salt_pose)
    pepper_state = Objstate("Pepper",
                            pose=init_pepper_pose)
    
    computer_poses = [(5,9), (6,2)]
    computer_states = []
    for pose in computer_poses:
        computer_states.append(Objstate("Computer",
                                        pose=pose))
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
    colors = {"Salt": (128, 128, 128),
              "Pepper": (200, 10, 10)}

    return ids, grid_map, init_state, colors


def office_floor2_1(init_robot_pose=(9,0,0)):
    """
    Office floor with salt, pepper, and computers
    This describes the full environment configuration (i.e. all
    object locations and the robot's location).
    """
    grid_map = all_maps["map_small_3"]()
    init_robot_pose = init_robot_pose

    # salt_id = 10
    # pepper_id = 15
    robot_id = 1
    robot_state = Objstate("Robot",
                           pose=init_robot_pose,
                           camera_direction="-x")
    # salt_state = Objstate("Salt",
    #                       pose=init_salt_pose)
    # pepper_state = Objstate("Pepper",
    #                         pose=init_pepper_pose)
    
    # computer_poses = [(5,9), (6,2)]
    # computer_states = []
    # for pose in computer_poses:
    #     computer_states.append(Objstate("Computer",
    #                                     pose=pose))
    init_state = {robot_id: robot_state}
                  # pepper_id: pepper_state,
                  # salt_id: salt_state}
    # for i, s in enumerate(computer_states):
    #     init_state[3000+i] = s

    # maps from object class to id        
    ids = {}  
    for objid in init_state:
        c = init_state[objid].objclass
        if c not in ids:
            ids[c] = []
        ids[c].append(objid)
    ids["Robot"] = ids["Robot"][0]
    # colors = {"Salt": (128, 128, 128),
    #           "Pepper": (200, 10, 10)}
    colors = {}
    return ids, grid_map, init_state, colors

