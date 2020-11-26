import argparse
import os
import pomdp_py
import pickle
import yaml
import copy
import time
import math
import subprocess
from relpomdp.home2d.constants import FILE_PATHS
from relpomdp.home2d.tests.test_utils import update_map
from relpomdp.oopomdp.framework import Objstate, OOState, OOBelief
from relpomdp.home2d.agent import NKAgentViz, Laser2DSensor, NKAgent, FakeSLAM
from relpomdp.home2d.utils import save_images_and_compress

def main():
    parser = argparse.ArgumentParser(description="replay a trial")
    parser.add_argument("trial_path", type=str, help="Path to trial directory")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--delay", type=float, help="delay in seconds between two steps")
    args = parser.parse_args()

    with open(os.path.join(args.trial_path, "trial.pkl"), "rb") as f:
        trial = pickle.load(f)

    agent_type = trial.config["agent_type"]

    env_file = trial.config["env_file"]
    env_id = trial.config["env_id"]
    env_path = os.path.join(FILE_PATHS["exp_worlds"], env_file)
    with open(env_path, "rb") as f:
        env = pickle.load(f)[env_id]

    with open(os.path.join(args.trial_path, "states.pkl"), "rb") as f:
        states = pickle.load(f)

    with open(os.path.join(args.trial_path, "history.pkl"), "rb") as f:
        history = pickle.load(f)

    with open(os.path.join(args.trial_path, "rewards.yaml"), "rb") as f:
        rewards = yaml.load(f, Loader=yaml.Loader)

    init_robot_pose = env.state.object_states[env.robot_id]["pose"]
    nk_agent = NKAgent(env.robot_id, init_robot_pose)

    slam_sensor_config = {}
    for sensor_name in trial.config["domain"]["sensors"]:
        cfg = trial.config["domain"]["sensors"][sensor_name]
        if sensor_name.lower().startswith("room"):
            slam_sensor_config = copy.deepcopy(cfg)
    fake_slam = FakeSLAM(Laser2DSensor(env.robot_id,
                                       fov=slam_sensor_config.get("fov", 90),
                                       min_range=slam_sensor_config.get("min_range", 1),
                                       max_range=slam_sensor_config.get("max_range", 3),
                                       angle_increment=slam_sensor_config.get("angle_increment", 0.1)))

    # We simulate that the robot first stays in place to map what's in front,
    # and then rotates 90 degrees to map what's on top. This gives the robot
    # a small initial map to work on instead of just its grid cell.
    update_map(fake_slam, nk_agent, init_robot_pose, init_robot_pose, env)
    rotated_robot_pose = (init_robot_pose[0], init_robot_pose[1], init_robot_pose[2] + math.pi/2)
    update_map(fake_slam, nk_agent, init_robot_pose, rotated_robot_pose, env)

    # Create visualization
    _game_states = []
    with open(FILE_PATHS["colors"]) as f:
        colors = yaml.load(f)
        for objclass in colors:
            colors[objclass] = pomdp_py.util.hex_to_rgb(colors[objclass])
    viz = NKAgentViz(nk_agent,
                     env,
                     colors,
                     res=30,
                     controllable=True,
                     img_path=FILE_PATHS["object_imgs"])
    viz.on_init()
    _, _, img = viz.on_render()
    _game_states.append(img)

    _gamma = 1.0
    _discount_factor = trial.config["planning"]["discount_factor"]
    _disc_reward = 0.0
    target_id = list(env.ids_for(trial.config["target_class"]))[0]
    for i in range(len(history)):
        start_time = time.time()
        if len(history[i]) == 2:
            action, observation = history[i]
            belief = None
        else:
            action, observation, belief = history[i]
            if isinstance(belief, dict):
                belief = OOBelief(belief)

        prev_robot_pose = env.state.object_states[env.robot_id]["pose"]
        _ = env.state_transition(action, execute=True,
                                 robot_id=env.robot_id)
        reward = rewards[i]
        robot_pose = env.state.object_states[env.robot_id]["pose"]
        update_map(fake_slam, nk_agent, prev_robot_pose, robot_pose, env)

        _disc_reward += reward * _gamma
        _gamma *= _discount_factor
        _step_info = "Step {} : Action: {}    Reward: {}    DiscCumReward: {:.4f}    RobotPose: {}   TargetFound: {}"\
            .format(i+1, action, reward, _disc_reward,
                    robot_pose,
                    env.state.object_states[target_id].get("is_found", False))

        viz.on_loop()
        _, _, img = viz.on_render(belief)
        print(_step_info)
        _game_states.append(img)

        step_time = time.time() - start_time
        if args.delay is not None and step_time < args.delay:
            time.sleep(args.delay - step_time)


    viz.on_cleanup()

    if args.save:
        print("Saving images...")
        dirp = args.trial_path
        save_images_and_compress(_game_states,
                                 dirp)
        subprocess.Popen(["nautilus", dirp])


if __name__ == "__main__":
    main()
