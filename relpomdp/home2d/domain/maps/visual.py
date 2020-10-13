# Visualize a map
from relpomdp.home2d.domain.visual import Home2DViz
# from relpomdp.home2d.domain.maps.prebuild import *
from relpomdp.home2d.domain.maps.build_map import pcg_map, pcg_world
from relpomdp.oopomdp.framework import Objstate
from relpomdp.home2d.domain.env import Home2DEnvironment
import random

def main():#map_func):
    grid_map = pcg_map(10, 10, 4, ["Office", "Kitchen", "Office", "Office"],
                       min_room_size=2, max_room_size=6, seed=random.randint(0, 100),
                       ndoors=2)
    init_state = pcg_world(grid_map,
                           {"Office": {
                               "Computer": (1, (1,1)),
                               "shelf": (1, (1,1)),
                           },
                            "Kitchen": {
                                "Oven": (1, (1,1)),
                                "Salt": (1, (1,1)),
                                "Pepper": (1, (1,1)),
                                "Pan": (1, (1,1))
                            }})
    robot_id = 0 # not important
    robot_state = Objstate("Robot",
                           pose=(0,0,0),
                           camera_direction="-x")  # not important    
    init_state[robot_id] = robot_state  # not important
    colors = {}
    env = Home2DEnvironment(robot_id,
                            grid_map,
                            init_state)
    viz = Home2DViz(env,
                    colors,
                    res=30,
                    img_path="../imgs")
    viz.on_init()
    img = viz.on_render()
    input("PRESS ENTER TO EXIT")

if __name__ == "__main__":
    main()#office_floor2_1)
