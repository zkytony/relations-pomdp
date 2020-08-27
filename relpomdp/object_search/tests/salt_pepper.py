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

    belief = {10: salt_hist,
              15: pepper_hist}

    init_belief = pomdp_py.Histogram(belief)
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
    viz.update(belief)

    if viz.on_init() == False:
        viz._running = False

    while( viz._running ):
        for event in pygame.event.get():
            action = viz.on_event(event)

            if action is not None:
                reward = env.state_transition(action, execute=True)
                print("robot state: %s" % str(env.robot_state))
                print("     action: %s" % str(action.name))
                print("     reward: %s" % str(reward))
                
                observation = agent.observation_model.sample(env.state, action)
                print("observation: %s" % str(observation))
                print("------------")

                # Update belief
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
                        belief = {10: salt_hist,
                                  15: pepper_hist}
                        agent.set_belief(pomdp_py.Histogram(belief))
                        viz.update(belief)
                break
            
        viz.on_loop()
        viz.on_render()
    viz.on_cleanup()
    
    
    viz.on_execute()


if __name__ == "__main__":
    main()
