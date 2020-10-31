from relpomdp.home2d.domain.maps.build_map import random_world
from relpomdp.home2d.domain.env import Home2DEnvironment
import argparse
import yaml
import pickle
import random
import os


def generate_world(config, seed=None):
    """Generates a single room"""
    # Obtain parameters
    # per-world configs
    width = config["width"]
    length = config["length"]
    nrooms = config["nrooms"]
    room_types = config["room_types"]
    objects_cfg = config["objects"]
    objects = {}
    for room_type in objects_cfg:
        objects[room_type] = {}
        for objclass in objects_cfg[room_type]:
            counts = objects_cfg[room_type].get("counts", 1)
            dims = objects_cfg[room_type].get("dims", (1,1))
            objects[room_type][objclass] = (counts, dims)

    robot_id = config.get("robot_id", 0)
    init_robot_pose = tuple(config.get("init_robot_pose", (0, 0, 0)))
    ndoors = config["ndoors"]
    min_room_size = config.get("min_room_size", 2)
    max_room_size = config.get("max_room_size", 6)
    shuffle_rooms = config["shuffle_rooms"]

    # Generate environments
    if shuffle_rooms:
        random.shuffle(room_types)
    init_state, grid_map = random_world(width, length, nrooms,
                                        room_types,
                                        objects=objects,
                                        robot_id=robot_id,
                                        init_robot_pose=init_robot_pose,
                                        ndoors=ndoors,
                                        min_room_size=min_room_size,
                                        max_room_size=max_room_size,
                                        seed=seed)
    env = Home2DEnvironment(robot_id,
                            grid_map,
                            init_state)
    return env


def main():
    parser = argparse.ArgumentParser(description="Generate random worlds")
    parser.add_argument("name",
                        type=str, help="Name of this batch of worlds")
    parser.add_argument("config_file",
                        type=str, help="Path to .yaml configuration file")
    parser.add_argument("output_dir",
                        type=str, help="Directory to output generated worlds")

    args = parser.parse_args()
    with open(args.config_file) as f:
        config = yaml.load(f)

    # # Obtain parameters
    # # per-world configs
    # width = config["width"]
    # length = config["length"]
    # nrooms = config["nrooms"]
    # room_types = config["room_types"]
    # objects_cfg = config["objects"]
    # objects = {}
    # for room_type in objects_cfg:
    #     objects[room_type] = {}
    #     for objclass in objects_cfg[room_type]:
    #         counts = objects_cfg[room_type].get("counts", 1)
    #         dims = objects_cfg[room_type].get("dims", (1,1))
    #         objects[room_type][objclass] = (counts, dims)

    # robot_id = config.get("robot_id", 0)
    # init_robot_pose = tuple(config.get("init_robot_pose", (0, 0, 0)))
    # ndoors = config["ndoors"]
    # min_room_size = config.get("min_room_size", 2)
    # max_room_size = config.get("max_room_size", 6)

    # generation configs
    num_worlds = config["num_worlds"]
    # shuffle_rooms = config["shuffle_rooms"]

    # Generate environments
    envs = {}  # Maps from environment ID to environment
    for i in range(num_worlds):
        print("Generating world %d" % i)
        envs[i] = generate_world(config)
        # if shuffle_rooms:
        #     random.shuffle(room_types)
        # init_state, grid_map = random_world(width, length, nrooms,
        #                                     room_types,
        #                                     objects=objects,
        #                                     robot_id=robot_id,
        #                                     init_robot_pose=init_robot_pose,
        #                                     ndoors=ndoors,
        #                                     min_room_size=min_room_size,
        #                                     max_room_size=max_room_size)
        # env = Home2DEnvironment(robot_id,
        #                         grid_map,
        #                         init_state)
        # envs[i] = env

    # Save the map as a tuple file
    with open(os.path.join(args.output_dir, "%s.pkl" % args.name), "wb") as f:
        pickle.dump(envs, f)
    print("Successfully generated %d worlds" % num_worlds)

if __name__ == "__main__":
    main()
