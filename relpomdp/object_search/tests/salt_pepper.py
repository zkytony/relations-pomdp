from relpomdp.object_search.world_specs.build_world import *
from relpomdp.object_search.env import *
from relpomdp.object_search.relation import *
import networkx as nx
import matplotlib.pyplot as plt
from pgmpy.inference import BeliefPropagation

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
    print(mrf.query(variables=["Salt_Pose"],
                    evidence={"Pepper_Pose": (5,5)}))
    
    print("Creating visualization ...")    
    viz = ObjectSearchViz(env,
                          {10: (128, 128, 128),
                           15: (200, 10, 10)},
                          res=30,
                          controllable=True,
                          img_path="../imgs")
    viz.on_init()
    viz.on_execute()


if __name__ == "__main__":
    main()
