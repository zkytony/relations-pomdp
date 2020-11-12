# Computes difficulty of searching for a class of object,
# including room type, by picking unexplored locations
# and move the robot there, recording observations received
# in the mean time.

import pomdp_py
from relpomdp.home2d.agent import NKAgentViz, Laser2DSensor, NKAgent, FakeSLAM
from relpomdp.home2d.agent.transition_model import DeclareFound
from relpomdp.home2d.learning.random_explorer import RandomPlanner
from relpomdp.oopomdp.framework import Objstate
from relpomdp.home2d.tests.test_utils import random_policy_model
from relpomdp.home2d.learning.correlation_observation_model\
    import compute_detections, CorrelationObservationModel
import argparse
import pickle
import yaml
import os
import copy
from datetime import datetime as dt

def build_training_agent(env, sensor_configs):
    robot_id = env.robot_id
    init_robot_pose = env.robot_state["pose"]
    nk_agent = NKAgent(robot_id, init_robot_pose, grid_map=env.grid_map)

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
    return nk_agent


def run_single(env, sensor_configs, nsteps=100, visualize=True):
    """
    Given an environment, deploy an agent that randomly moves in this environment.
    The `sensors` specify the sensor parameters and uncertainty when detecting different objects;

    It maps from a class_name to a (alpha, beta) tuple.

    `sensors` is a dictionary, {sensor_name -> {'fov':FOV, 'min_range':min_range, 'max_range':max_range,
                                                'noises': {object_class: [alpha, beta]}}}
    """
    nk_agent = build_training_agent(env, sensor_configs)
    policy_model = random_policy_model(nk_agent)
    agent = nk_agent.instantiate(policy_model)
    planner = RandomPlanner(env.robot_id, env.legal_motions)

    # Visualize and run
    if visualize:
        viz = NKAgentViz(agent, env, {},
                         res=30,
                         controllable=True,
                         img_path="../domain/imgs")
        viz.on_init()

    all_detections = []
    for i in range(nsteps):
        # Visualize
        if visualize:
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
        detected_classes, detected_ids, detected_poses = compute_detections(observation, return_poses=True)
        all_detections.append((detected_classes, detected_ids, detected_poses))

        # update belief (only need to do so for robot belief)
        agent.belief.object_beliefs[env.robot_id] = pomdp_py.Histogram({
            env.robot_state.copy() : 1.0
        })
    if visualize:
        viz.on_cleanup()
    return all_detections


def main():
    parser = argparse.ArgumentParser(description="Simulates a random agent running in the world")
    parser.add_argument("config_file",
                        type=str, help="Path to .yaml configuration file")
    parser.add_argument("path_to_envs",
                        type=str, help="Path to a .pickle file that contains a collection of environments")
    parser.add_argument("output_dir",
                        type=str, help="Directory to detections saved in a file")
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
            detections[envid] = run_single(envs[envid], config["sensors"], nsteps=args.nsteps, visualize=False)
            if len(detections) >= args.trials:
                break
    finally:
        with open(os.path.join(args.output_dir, "detections-random-%d-%s-%s.pkl" % (args.nsteps, filename, timestr)), "wb") as f:
            pickle.dump(detections, f)


if __name__ == "__main__":
    main()
