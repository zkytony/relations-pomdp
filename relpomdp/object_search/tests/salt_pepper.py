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
    Office floor with many rooms. There is a kitchen,
    a coffee room, and other rooms are offices. There is a computer in
    every office. There's salt and pepper in the kitchen.

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
    mrf_path = os.path.join(mrfdir, "salt_pepper_1.pkl")
    if os.path.exists(mrf_path):
        with open(mrf_path, "rb") as f:
            mrf = pickle.load(f)
    else:
        near_salt_pepper = Near("Salt", "Pepper", grid_map)  # grounding the relation on the grid map
        near_salt_computer = Near("Salt", "Computer", grid_map, negate=True)
        mrf = relations_to_mrf([near_salt_pepper, near_salt_computer])
        if save_mrf:
            os.makedirs(mrfdir, exist_ok=True)
            with open(mrf_path, "wb") as f:
                pickle.dump(mrf, f)
    
    colors = {"Salt": (128, 128, 128),
              "Pepper": (200, 10, 10)}
    return ids, grid_map, init_state, mrf, colors


def main(world=office_floor1):

    # Build a map
    print("Creating map ...")
    ids, grid_map, init_state, relations, colors = world()
    
    print("Creating environment ...")
    env = ObjectSearchEnvironment(ids,
                                  grid_map,
                                  init_state)
    mrf = relations_to_mrf(relations)

    # The mrf is simply the initial belief. Just plot it by objects.
    salt_phi = mrf.query(variables=["Salt_pose"])
    salt_phi.normalize()
    salt_hist_mrf = {}
    for loc in mrf.values("Salt_pose"):
        state = ItemState("Salt", loc)
        salt_hist_mrf[state] = salt_phi.get_value({"Salt_pose":loc})

    salt_id = ids["Salt"][0]
    robot_id = ids["Robot"]
    init_belief = pomdp_py.OOBelief({salt_id: pomdp_py.Histogram(salt_hist_mrf),
                                     robot_id: pomdp_py.Histogram({env.robot_state:1.0})})
    sensor = Laser2DSensor(robot_id, env.grid_map, fov=90, min_range=1, max_range=2,
                           angle_increment=0.5)
    agent = ObjectSearchAgent(env.grid_map, sensor, env.ids,
                              init_belief)
    
    print("Creating visualization ...")
    objcolors = {}
    for objid in env.state.object_states:
        s_o = env.state.object_states[objid]
        if s_o.objclass in colors:
            objcolors[objid] = colors[s_o.objclass]
    viz = ObjectSearchViz(env,
                          objcolors,
                          res=30,
                          controllable=True,
                          img_path="../imgs")
    viz.on_init()
    viz.on_render()    
    viz.update({10:salt_hist_mrf})
    viz.on_render()        

    planner = pomdp_py.POUCT(max_depth=20,
                             discount_factor=0.95,
                             num_sims=200,
                             exploration_const=100,
                             rollout_policy=agent.policy_model)
    # planner = GreedyPlanner(ids)

    used_objects = set()  # objects who has contributed to mrf belief update    
    for step in range(100):
        # # Input action from keyboard
        # action = None
        # while action is None:
        #     for event in pygame.event.get():
        #         if event.type == pygame.KEYDOWN:
        #             action = viz.on_event(event)
        #     time.sleep(0.1)
        print("---- Step %d ----" % step)        

        action = planner.plan(agent)
        # time.sleep(0.1)
        print("   num sims: %d" % planner.last_num_sims)

        robot_state = env.robot_state.copy()
        reward = env.state_transition(action, execute=True)
        print("robot state: %s" % str(env.robot_state))
        print("     action: %s" % str(action.name))
        print("     reward: %s" % str(reward))

        observation = agent.observation_model.sample(env.state, action)
        print("observation: %s" % str(observation))

        # Compute a distribution using the MRF
        updating_mrf = False
        for objid in observation.object_observations:
            o_obj = observation.object_observations[objid]
            if objid != env.ids["Robot"]\
               and objid not in used_objects:
                objclass = o_obj.objclass
                if objclass == "Salt":
                    # You just observed Salt. MRF isn't useful here.
                    continue
                
                pose = o_obj.pose

                full_phi = mrf.query(variables=["Salt_pose"],
                                     evidence={"%s_pose" % objclass: pose})
                salt_phi = full_phi#.marginalize(["Pepper_pose"], inplace=False)
                salt_phi.normalize()
                for loc in mrf.values("Salt_pose"):
                    state = ItemState("Salt", loc)
                    salt_hist_mrf[state] = salt_phi.get_value({"Salt_pose":loc})
                updating_mrf = True
                used_objects.add(objid)

        # Compute a distribution using the standard belief update
        current_salt_hist = agent.belief.object_beliefs[salt_id]
        next_robot_state = env.robot_state.copy()
        
        new_histogram = {}  # state space still the same.
        total_prob = 0
        for next_salt_state in current_salt_hist:
            next_state = JointState({salt_id: next_salt_state,
                                     robot_id: next_robot_state})
            observation_prob = agent.observation_model.probability(
                observation.for_objs([robot_id,salt_id]), next_state, action)
            mrf_prob = 1.0
            if updating_mrf:
                mrf_prob = salt_hist_mrf[next_salt_state]
            
            transition_prob = current_salt_hist[next_salt_state]
            new_histogram[next_salt_state] = mrf_prob * observation_prob * transition_prob
            total_prob += new_histogram[next_salt_state]

        # Normalize
        for salt_state in new_histogram:
            if total_prob > 0:
                new_histogram[salt_state] /= total_prob
        salt_hist_update = new_histogram

        agent.set_belief(pomdp_py.OOBelief({salt_id:pomdp_py.Histogram(salt_hist_update),
                                            robot_id:pomdp_py.Histogram({env.robot_state:1.0})}))
        planner.update(agent, action, observation)
        viz.update({salt_id: salt_hist_update})
        viz.on_loop()
        viz.on_render()

        # Terminates
        if env.state.object_states[salt_id].is_found:
            break
        
    viz.on_cleanup()

if __name__ == "__main__":
    main()
