from relpomdp.object_search.world_specs.build_world import *
from relpomdp.object_search.env import *
from relpomdp.object_search.agent import *
from relpomdp.object_search.relation import *
import networkx as nx
import matplotlib.pyplot as plt
from pgmpy.inference import BeliefPropagation
from search_and_rescue.experiments.plotting import *
import pomdp_py
import pygame

def main():
    # Build a world
    print("Creating world ...")    
    grid_map = small_world0() # 10 by 10 world

    # Arbitrary states
    robot_state = RobotState((1,4), "-x")
    salt_state = ItemState("Salt", (0,1))
    pepper_state = ItemState("Pepper", (1,3))
    init_state = {1: robot_state,
                  10: salt_state,
                  15: pepper_state}
    print("Creating environment ...")
    env = ObjectSearchEnvironment(grid_map,
                                  init_state,
                                  {10})

    near_salt_pepper = Near("Salt", "Pepper", env.grid_map)
    mrf = near_salt_pepper.to_mrf()

    # The mrf is simply the initial belief. Just plot it
    # by objects.
    full_phi = mrf.query(variables=["Salt_Pose",
                                    "Pepper_Pose"])
    salt_phi = full_phi.marginalize(["Pepper_Pose"], inplace=False)
    salt_phi.normalize()
    pepper_phi = full_phi.marginalize(["Salt_Pose"], inplace=False)
    pepper_phi.normalize()
    
    salt_hist_mrf = {}
    pepper_hist_mrf = {}
    for loc in mrf.values("Salt_Pose"):
        state = ItemState("Salt", loc)
        salt_hist_mrf[state] = salt_phi.get_value({"Salt_Pose":loc})
    for loc in mrf.values("Pepper_Pose"):
        state = ItemState("Pepper", loc)
        pepper_hist_mrf[state] = pepper_phi.get_value({"Pepper_Pose":loc})        

    init_belief = pomdp_py.OOBelief({10: pomdp_py.Histogram(salt_hist_mrf),
                                     1: pomdp_py.Histogram({robot_state:1.0})})
    agent = ObjectSearchAgent(env.grid_map, env.ids,
                              init_belief)
    
    print("Creating visualization ...")    
    viz = ObjectSearchViz(env,
                          {10: (128, 128, 128),
                           15: (200, 10, 10)},
                          res=30,
                          controllable=True,
                          img_path="../imgs")
    viz.on_init()
    viz.on_render()    
    viz.update({10:salt_hist_mrf})
    viz.on_render()        

    planner = pomdp_py.POUCT(max_depth=30,
                             discount_factor=0.95,
                             num_sims=150,
                             exploration_const=2,
                             rollout_policy=agent.policy_model)  # Random by default    

    used_objects = set()  # objects who has contributed to mrf belief update    
    for step in range(100):
        # Input action from keyboard
        # for event in pygame.event.get():
        #     action = viz.on_event(event)
        print("---- Step %d ----" % step)        

        action = planner.plan(agent)
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
               and objid not in used_objects\
               and o_obj.objclass in {"Salt", "Pepper"}:
                objclass = o_obj.objclass
                pose = o_obj.pose

                query_vars = ["%s_Pose" % c for c in {"Salt", "Pepper"} - {objclass}]
                full_phi = mrf.query(variables=query_vars,
                                     evidence={"%s_Pose" % objclass: pose})

                # TODO: REFACTROR                        
                papper_hist = {}
                salt_hist_mrf = {}                        
                if objclass == "Salt":
                    pepper_phi = full_phi#.marginalize(["Salt_Pose"], inplace=False)
                    pepper_phi.normalize()
                    for loc in mrf.values("Pepper_Pose"):
                        state = ItemState("Pepper", loc)
                        pepper_hist_mrf[state] = pepper_phi.get_value({"Pepper_Pose":loc})
                        salt_hist_mrf[ItemState("Salt", loc)] = 1e-9
                    salt_hist_mrf[ItemState("Salt", pose)] = 1.0 - 1e-9

                else:
                    salt_phi = full_phi#.marginalize(["Pepper_Pose"], inplace=False)
                    salt_phi.normalize()
                    for loc in mrf.values("Salt_Pose"):
                        state = ItemState("Salt", loc)
                        salt_hist_mrf[state] = salt_phi.get_value({"Salt_Pose":loc})
                        pepper_hist_mrf[ItemState("Pepper", loc)] = 1e-9                        
                    pepper_hist_mrf[ItemState("Pepper", pose)] = 1.0
                updating_mrf = True
                used_objects.add(objid)

        # Compute a distribution using the standard belief update
        current_salt_hist = agent.belief.object_beliefs[10]
        next_robot_state = env.robot_state.copy()
        
        new_histogram = {}  # state space still the same.
        total_prob = 0
        for next_salt_state in current_salt_hist:
            next_state = JointState({10: next_salt_state,
                                     1: next_robot_state})
            observation_prob = agent.observation_model.probability(
                observation.for_objs([1,10]), next_state, action)
            mrf_prob = 1.0
            if updating_mrf:
                mrf_prob = salt_hist_mrf[next_salt_state]
            
            transition_prob = current_salt_hist[next_salt_state]
            new_histogram[next_salt_state] = mrf_prob * observation_prob * transition_prob

            if updating_mrf:
                if next_salt_state.pose == pepper_state.pose:
                    print("Pepper!", new_histogram[next_salt_state])
                else:
                    print("---", new_histogram[next_salt_state])
                print("        T", transition_prob, "  O", observation_prob,  "  M", mrf_prob)
            else:
                if next_salt_state.pose == pepper_state.pose:
                    print("Pepper!", new_histogram[next_salt_state])
                    print("        T", transition_prob, "  O", observation_prob,  "  M", mrf_prob)                    
                    
            total_prob += new_histogram[next_salt_state]

        # Normalize
        for salt_state in new_histogram:
            if total_prob > 0:
                new_histogram[salt_state] /= total_prob
        salt_hist_update = new_histogram

        agent.set_belief(pomdp_py.OOBelief({10:pomdp_py.Histogram(salt_hist_update),
                                            1:pomdp_py.Histogram({env.robot_state:1.0})}))
        planner.update(agent, action, observation)
        viz.update({10: salt_hist_update})
        viz.on_loop()
        viz.on_render()

        # Terminates
        if env.state.object_states[10].is_found:
            break
        
    viz.on_cleanup()

if __name__ == "__main__":
    main()
