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
    grid_map = small_world1() # 10 by 10 world

    # Arbitrary states
    robot_state = RobotState((9,6), "-x")
    salt_state = ItemState("Salt", (0,6))
    pepper_state = ItemState("Pepper", (1,4))
    init_state = {1: robot_state,
                  10: salt_state,
                  15: pepper_state}
    print("Creating environment ...")
    env = ObjectSearchEnvironment(grid_map,
                                  init_state,
                                  {10})

    near_salt_pepper = Near("Salt", "Pepper", env.grid_map)
    mrf = near_salt_pepper.to_mrf()
    phi = mrf.query(variables=["Salt_Pose"],
                    evidence={"Pepper_Pose": (5,5)})

    # The mrf is simply the initial belief. Just plot it
    # by objects.
    full_phi = mrf.query(variables=["Salt_Pose",
                                    "Pepper_Pose"])
    salt_phi = full_phi.marginalize(["Pepper_Pose"], inplace=False)
    salt_phi.normalize()
    pepper_phi = full_phi.marginalize(["Salt_Pose"], inplace=False)
    pepper_phi.normalize()
    
    salt_hist = {}
    pepper_hist = {}
    for loc in mrf.values("Salt_Pose"):
        state = ItemState("Salt", loc)
        salt_hist[state] = salt_phi.get_value({"Salt_Pose":loc})
    for loc in mrf.values("Pepper_Pose"):
        state = ItemState("Pepper", loc)
        pepper_hist[state] = pepper_phi.get_value({"Pepper_Pose":loc})        

    init_belief = pomdp_py.OOBelief({10: pomdp_py.Histogram(salt_hist),
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
    viz.update({10:salt_hist})
    viz.on_render()        

    planner = pomdp_py.POUCT(max_depth=10,
                             discount_factor=0.95,
                             num_sims=100,
                             exploration_const=2,
                             rollout_policy=agent.policy_model)  # Random by default    
        
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
        for objid in observation.object_observations:
            o_obj = observation.object_observations[objid]
            if objid != env.ids["Robot"]\
               and o_obj.objclass in {"Salt", "Pepper"}:
                objclass = o_obj.objclass
                pose = o_obj.pose

                query_vars = ["%s_Pose" % c for c in {"Salt", "Pepper"} - {objclass}]
                full_phi = mrf.query(variables=query_vars,
                                     evidence={"%s_Pose" % objclass: pose})

                # TODO: REFACTROR                        
                papper_hist = {}
                salt_hist = {}                        
                if objclass == "Salt":
                    pepper_phi = full_phi#.marginalize(["Salt_Pose"], inplace=False)
                    pepper_phi.normalize()
                    for loc in mrf.values("Pepper_Pose"):
                        state = ItemState("Pepper", loc)
                        pepper_hist[state] = pepper_phi.get_value({"Pepper_Pose":loc})
                    salt_hist[ItemState("Salt", pose)] = 1.0

                else:
                    salt_phi = full_phi#.marginalize(["Pepper_Pose"], inplace=False)
                    salt_phi.normalize()
                    for loc in mrf.values("Salt_Pose"):
                        state = ItemState("Salt", loc)
                        salt_hist[state] = salt_phi.get_value({"Salt_Pose":loc})
                    pepper_hist[ItemState("Pepper", pose)] = 1.0
        salt_hist_mrf = salt_hist

        # Compute a distribution using the standard belief update
        current_salt_hist = agent.belief.object_beliefs[10]
        next_robot_state = env.robot_state.copy()
        
        new_histogram = {}  # state space still the same.
        total_prob = 0
        for next_salt_state in current_salt_hist:
            next_state = JointState({10: next_salt_state,
                                     1: next_robot_state})
            observation_prob = agent.observation_model.probability(observation,
                                                                   next_state,
                                                                   action)
            transition_prob = current_salt_hist[next_salt_state]
            new_histogram[next_salt_state] = observation_prob * transition_prob
            total_prob += new_histogram[next_salt_state]

        # Normalize
        for salt_state in new_histogram:
            if total_prob > 0:
                new_histogram[salt_state] /= total_prob
        salt_hist_update = new_histogram

        # Multiply the two
        salt_hist = {}
        for state in salt_hist_mrf:
            salt_hist[state] = salt_hist_mrf[state] * salt_hist_update[state]
        agent.set_belief(pomdp_py.OOBelief({10:pomdp_py.Histogram(salt_hist_mrf),
                                            1:pomdp_py.Histogram({env.robot_state:1.0})}))
        planner.update(agent, action, observation)        
        viz.update({10: salt_hist_mrf})
        

        viz.on_loop()
        viz.on_render()
    viz.on_cleanup()

if __name__ == "__main__":
    main()
