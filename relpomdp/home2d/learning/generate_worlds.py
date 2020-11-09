from relpomdp.home2d.domain.maps.build_map import random_world
from relpomdp.home2d.domain.env import Home2DEnvironment
from relpomdp.oopomdp.framework import Objstate
from relpomdp.oopomdp.graph import Graph, Edge, Node
from relpomdp.home2d.domain.condition_effect import MoveEffect
import argparse
import yaml
import pickle
import random
import os

def add_room_states(env, starting_room_id=10000):
    """Given environemnt, add in its state space a
    state for each room, which is located at one of the room's doorways"""
    # We will add a state per doorway per room
    room_id = starting_room_id
    for room_name in env.grid_map.rooms:
        room = env.grid_map.rooms[room_name]
        for doorway in room.doorways:
            room_state = Objstate(room.room_type,
                                  pose=doorway)
            env.add_object_state(room_id, room_state)
            room_id += 100
        room_id += 1000

def _coverable(env):
    """Returns true if all locations in the environment could be visited."""
    # We can actually make use of the Graph, just so we don't reimplement
    # the same code.
    nodes = {}
    edges = set()
    for x in range(env.grid_map.width):
        for y in range(env.grid_map.length):
            nid = (x,y)  # use location as id
            nodes[nid] = Node(nid)
            # Get legal motions, compute neigbors
            for motion in env.legal_motions[(x,y)]:
                neighbor_nid = MoveEffect.move_by((x,y), motion)[:2]
                if neighbor_nid not in nodes:
                    nodes[neighbor_nid] = Node(neighbor_nid)
                edge = Edge(len(edges), nodes[nid], nodes[neighbor_nid], data=motion)
                edges.add(edge)
    graph = Graph(edges)
    components = graph.connected_components()
    return len(components) == 1


def _has_required_classes(env, required_classes):
    for c in required_classes:
        if len(env.ids_for(c)) == 0:
            return False
    return True

def _generate_world_helper(config, seed=None, required_classes=set()):
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
    add_room_states(env)
    return env

def generate_world(config, seed=None, required_classes=set()):
    while True:
        env = _generate_world_helper(config, seed=seed, required_classes=required_classes)
        has_req_c = _has_required_classes(env, required_classes)
        coverable = _coverable(env)
        if has_req_c and coverable:
            break
        else:
            if not has_req_c:
                print("Generated map does not contain required classes {}. Regenerate"\
                      .format(required_classes))
            if not coverable:
                print("Generated map is not fully coverable. Regenerate.")
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

    # generation configs
    num_worlds = config["num_worlds"]

    # Generate environments
    envs = {}  # Maps from environment ID to environment
    for i in range(num_worlds):
        print("Generating world %d" % i)
        envs[i] = generate_world(config)

    # Save the map as a tuple file
    with open(os.path.join(args.output_dir, "%s.pkl" % args.name), "wb") as f:
        pickle.dump(envs, f)
    print("Successfully generated %d worlds" % num_worlds)

if __name__ == "__main__":
    main()
