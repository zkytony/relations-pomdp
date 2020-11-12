# Runs a pomdp searcher to search for each object category
# that appears in an environment. Record the step that
# the searcher detects the object.

import pomdp_py
from relpomdp.home2d.agent import NKAgentViz, Laser2DSensor, NKAgent, FakeSLAM
from relpomdp.home2d.agent.transition_model import DeclareFound
from relpomdp.home2d.learning.random_explorer import RandomPlanner
from relpomdp.oopomdp.framework import Objstate
from relpomdp.home2d.tests.test_utils import random_policy_model
from relpomdp.home2d.learning.correlation_observation_model\
    import compute_detections, CorrelationObservationModel
from relpomdp.home2d.tests.test_pomdp_nk import test_pomdp_nk
import argparse
import pickle
import yaml
import os
import copy
from datetime import datetime as dt

def run_single(env, sensor_configs, nsteps=100, visualize=True,
               discount_factor=0.95, max_depth=20, num_sims=600,
               exploration_constant=200):
    """For each target class that can be handled, run a pomdp_nk
    agent. Record the detections (which is in the returned history)"""
    robot_id = env.robot_id
    init_robot_pose = env.robot_state["pose"]

    slam_sensor_config = {}
    for sensor_name in sensor_configs:
        cfg = sensor_configs[sensor_name]
        noises = cfg["noises"]
        if sensor_name.startswith("room"):
            slam_sensor_config = cfg

    detections = []
    for sensor_name in sensor_configs:
        cfg = sensor_configs[sensor_name]
        sensor = Laser2DSensor(robot_id,
                               name=sensor_name,
                               fov=float(cfg["fov"]),
                               min_range=float(cfg["min_range"]),
                               max_range=float(cfg["max_range"]),
                               angle_increment=float(cfg["angle_increment"]))
        noises = cfg["noises"]
        for target_class in noises:
            if len(env.ids_for(target_class)) > 0:
                target_sensor_config = copy.deepcopy(cfg)
                target_sensor_config["noises"] = cfg["noises"][target_class]
                env_copy = copy.deepcopy(env)
                rewards, states, history = test_pomdp_nk(env_copy, target_class,
                                                         discount_factor=discount_factor,
                                                         max_depth=max_depth,
                                                         num_sims=num_sims,
                                                         exploration_constant=exploration_constant,
                                                         nsteps=nsteps,
                                                         target_sensor_config=target_sensor_config,
                                                         slam_sensor_config=slam_sensor_config,
                                                         visualize=visualize)
                observation = history[-1][1]
                for objid in observation.object_observations:
                    objo = observation.object_observations[objid]
                    if objo["label"] == target_class:
                        detection_step = len(history)
                        detections.append((target_class, objid, detection_step))
    return detections

def main():
    parser = argparse.ArgumentParser(description="Simulates a random agent running in the world")
    parser.add_argument("config_file",
                        type=str, help="Path to .yaml configuration file")
    parser.add_argument("path_to_envs",
                        type=str, help="Path to a .pickle file that contains a collection of environments")
    parser.add_argument("output_dir",
                        type=str, help="Directory to output detections saved in a file")
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
            detections[envid] = run_single(envs[envid], config["sensors"], nsteps=args.nsteps,
                                           num_sims=1000, visualize=False)
            if len(detections) >= args.trials:
                break
    finally:
        with open(os.path.join(args.output_dir, "detections-pomdp-%d-%s-%s.pkl" % (args.nsteps, filename, timestr)), "wb") as f:
            pickle.dump(detections, f)


if __name__ == "__main__":
    main()
