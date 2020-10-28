# Computes difficulty of searching for a class of object,
# including room type, by picking unexplored locations
# and move the robot there, recording observations received
# in the mean time.

import pomdp_py
from relpomdp.home2d.agent.tests.test_fake_slam import wait_for_action
from relpomdp.home2d.agent.nk_agent import NKAgent, FakeSLAM
from relpomdp.home2d.tasks.common.sensor import Laser2DSensor
from relpomdp.home2d.agent.visual import NKAgentViz
from relpomdp.home2d.domain.maps.build_map import random_world
from relpomdp.home2d.domain.env import Home2DEnvironment
from relpomdp.home2d.agent.transition_model import Pickup
from relpomdp.home2d.learning.random_explorer import RandomPlanner
from relpomdp.oopomdp.framework import Objstate
from relpomdp.home2d.agent.tests.test_utils import random_policy_model
import argparse
import pickle
import yaml
import os
import copy
from datetime import datetime as dt


def run_single(env, sensor_configs, nsteps=100):
    """
    Given an environment, deploy an agent that randomly moves in this environment.
    The `sensors` specify the sensor parameters and uncertainty when detecting different objects;

    It maps from a class_name to a (alpha, beta) tuple.

    `sensors` is a dictionary, {sensor_name -> {'fov':FOV, 'min_range':min_range, 'max_range':max_range,
                                                'noises': {object_class: [alpha, beta]}}}
    """
    robot_id = env.robot_id
    init_robot_pose = env.robot_state["pose"]
    nk_agent = NKAgent(robot_id, init_robot_pose, grid_map=env.grid_map)

    # We will add to the environment state grid cells for each room
    room_states = {}
    room_id = 2000
    for room_name in env.grid_map.rooms:
        room = env.grid_map.rooms[room_name]
        grid_id = 0
        for x, y in room.locations:
            s = Objstate(room.room_type, pose=(x,y))
            room_states[room_id + grid_id] = s
            grid_id += 1
        room_id += 100
    env.state.object_states.update(room_states)

    for sensor_name in sensor_configs:
        cfg = sensor_configs[sensor_name]
        sensor = Laser2DSensor(robot_id,
                               name=sensor_name,
                               fov=float(cfg["fov"]),
                               min_range=float(cfg["min_range"]),
                               max_range=float(cfg["max_range"]),
                               angle_increment=float(cfg["angle_increment"]))
        noises = cfg["noises"]
        nk_agent.add_sensor(sensor, noises)
    policy_model = random_policy_model(nk_agent)

    agent = nk_agent.instantiate(policy_model)

    planner = RandomPlanner(robot_id, env.legal_motions)

    # Visualize and run
    viz = NKAgentViz(agent, env, {},
                     res=30,
                     controllable=True,
                     img_path="../domain/imgs")
    viz.on_init()

    all_detections = []
    for i in range(nsteps):
        # Visualize
        viz.on_loop()
        viz.on_render()

        action = planner.plan(agent)

        # environment transitions and obtains reward (note that we use agent's reward model for convenience)
        env_state = env.state.copy()
        _ = env.state_transition(action, execute=True)
        env_next_state = env.state.copy()

        # Obtain observation
        observation = agent.observation_model.sample(env.state, action)

        # Filter observation to get detections
        detections = {}
        for o in observation.observations:
            for objid in o.object_observations:
                objo = o.object_observations[objid]
                if objo["label"] != "unknown"\
                   and objo["label"] != "free":
                    detections[objid] = objo
        print(detections)
        all_detections.append(detections)

        # update belief (only need to do so for robot belief)
        agent.belief.object_beliefs[robot_id] = pomdp_py.Histogram({
            env.robot_state.copy() : 1.0
        })
        if isinstance(action, Pickup):
            print("Done.")
            break
    viz.on_cleanup()
    return all_detections


def main():
    parser = argparse.ArgumentParser(description="Simulates a random agent running in the world")
    parser.add_argument("config_file",
                        type=str, help="Path to .yaml configuration file")
    parser.add_argument("path_to_envs",
                        type=str, help="Path to a .pickle file that contains a collection of environments")
    parser.add_argument("output_dir",
                        type=str, help="Directory to output computed difficulty saved in a file")
    parser.add_argument("-n", "--nsteps", default=100,
                        type=int, help="Number of steps to run the agent in each training environment")
    parser.add_argument("-T", "--trials", default=100,
                        type=int, help="Number of worlds to explore out of all worlds in the collection of environments")
    args = parser.parse_args()

    with open(args.path_to_envs, "rb") as f:
        envs = pickle.load(f)
    with open(args.config_file) as f:
        config = yaml.load(f)

    filename = os.path.splitext(os.path.basename(args.path_to_envs))[0]

    start_time = dt.now()
    timestr = start_time.strftime("%Y%m%d%H%M%S%f")[:-3]

    detections = {}
    try:
        for envid in envs:
            detections[envid] = run_single(envs[envid], config["sensors"], nsteps=args.nsteps)
            if len(detections) >= args.trials:
                break
    finally:
        with open(os.path.join(args.output_dir, "detections-%d-%s-%s.pkl" % (args.nsteps, filename, timestr)), "wb") as f:
            pickle.dump(detections, f)


if __name__ == "__main__":
    main()
