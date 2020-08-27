from relpomdp.object_search.world_specs.build_world import *
from relpomdp.object_search.env import *
import pickle

if __name__ == "__main__":
    # Build a world
    print("Creating world ...")    
    grid_map = big_world1(50, 50)

    # Arbitrary states
    robot_state = RobotState((0,0), "+x")
    salt_state = ItemState("Salt", (3,3))
    pepper_state = ItemState("Pepper", (3,2))
    init_state = {1: robot_state,
                  10: salt_state,
                  15: pepper_state}
    print("Creating environment ...")
    env = ObjectSearchEnvironment(grid_map,
                                  init_state,
                                  {10})
    print("Creating visualization ...")    
    viz = ObjectSearchViz(env,
                          {10: (128, 128, 128),
                           15: (200, 10, 10)},
                          res=15,
                          controllable=True,
                          img_path="../imgs")
    viz.on_init()
    viz.on_execute()
