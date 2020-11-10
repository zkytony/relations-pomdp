import argparse
import os
import pickle
import yaml
import copy
from relpomdp.home2d.constants import FILE_PATHS
from relpomdp.home2d.tests.test_utils import update_map
from relpomdp.oopomdp.framework import Objstate, OOState, OOBelief
from relpomdp.home2d.agent import NKAgentViz, Laser2DSensor, NKAgent, FakeSLAM

def main():
    parser = argparse.ArgumentParser(description="replay a trial")
    parser.add_argument("trial_path", type=str, help="Path to trial directory")
    parser.add_argument("--save", action="store_true")
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
    viz = NKAgentViz(nk_agent,
                     env,
                     {},
                     res=30,
                     controllable=True,
                     img_path=FILE_PATHS["object_imgs"])
    viz.on_init()

    for i in range(len(history)):
        viz.on_loop()

        if len(history) == 2:
            action, observation = history[i]
        else:
            action, observation, belief = history[i]
            if isinstance(belief, dict):
                belief = OOBelief(belief)
        viz.on_render(belief)

        prev_robot_pose = env.state.object_states[env.robot_id]["pose"]
        reward = env.state_transition(action, execute=True,
                                      robot_id=env.robot_id)
        robot_pose = env.state.object_states[env.robot_id]["pose"]
        update_map(fake_slam, nk_agent, prev_robot_pose, robot_pose, env)

    viz.on_cleanup()


if __name__ == "__main__":
    main()
