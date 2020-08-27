from relpomdp.object_search.world_specs.build_world import *
from relpomdp.object_search.env import *
from relpomdp.object_search.relation import *
import networkx as nx
import matplotlib.pyplot as plt
from pgmpy.inference import BeliefPropagation
from search_and_rescue.experiments.plotting import *

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
    
    print("Creating visualization ...")    
    viz = ObjectSearchViz(env,
                          {10: (128, 128, 128),
                           15: (200, 10, 10)},
                          res=30,
                          controllable=True,
                          img_path="../imgs")
    viz.on_init()
    viz.update(belief)
    viz.on_execute()


if __name__ == "__main__":
    main()
