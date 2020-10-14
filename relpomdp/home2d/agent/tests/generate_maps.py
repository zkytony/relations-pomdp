# Generates a bunch of maps
import os
from relpomdp.home2d.domain.maps.build_map import random_world
from relpomdp.home2d.domain.env import Home2DEnvironment
import random
import pickle

MAP_CONFIGS = [
    # world width, world length, num_rooms, min_dim, max_dim
    ((4, 4, 2, 2, 2), ["Kitchen", "Office"]),
    ((6, 6, 3, 2, 4), ["Kitchen", "Bathroom", "Office"]),
    ((10, 10, 3, 2, 6), ["Kitchen", "Bathroom", "Office"])
]

OBJECTS = {"Office": {"Computer": (1, (1,1))},
           "Kitchen": {"Pepper": (1, (1,1)),
                       "Salt": (1, (1,1))},
           "Bathroom": {"Toilet": (1, (1,1))}}

INIT_ROBOT_POSE = (0, 0, 0)
ROBOT_ID = 0

NUM_VARIATIONS = 10

def main(save_dir="test_maps"):
    os.makedirs(save_dir, exist_ok=True)

    envs = {}
    for cfg in MAP_CONFIGS:
        for i in range(NUM_VARIATIONS):
            seed = random.randint(0,100)
            scale, room_categories = cfg
            width, length, num_rooms, min_dim, max_dim = scale
            init_state, grid_map = random_world(width, length, num_rooms, room_categories,
                                                objects=OBJECTS,
                                                robot_id=ROBOT_ID,
                                                init_robot_pose=INIT_ROBOT_POSE,
                                                min_room_size=min_dim,
                                                max_room_size=max_dim,
                                                seed=seed)
            env = Home2DEnvironment(ROBOT_ID,
                                    grid_map,
                                    init_state)
            cfg_name = "%s#%s_%d" % ("-".join(map(str,scale)), "-".join(room_categories), seed)
            with open(os.path.join(save_dir, "%s.pkl" % cfg_name), "wb") as f:
                pickle.dump(env, f)
                print("Saved map %s" % cfg_name)

if __name__ == "__main__":
    main(save_dir="test_maps")
