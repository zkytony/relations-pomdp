from navigation import *
from relpomdp.realistic.utils.ai2thor_utils import save_frames
from relpomdp.realistic.environment import ThorEnv
import time
import os

def test_motion_sequence():
    def step_cb(event, env=None):
        print(env.agent_pose())

    motions = build_motion_actions()
    motions = {m.name:m for m in motions}

    config = {
        "scene_name": "FloorPlan30",
        "agent_mode": "default",
        "width": 400,
        "height": 400
    }
    env = ThorEnv(config)
    env.launch()

    seq = ["MoveAhead",
           "RotateLeft",
           "RotateLeft",
           "RotateRight",
           "RotateRight",
           "MoveBack",
           "RotateLeft",
           "MoveAhead",
           "MoveAhead"]
    actions = [motions[name].to_thor_action() for name in seq]
    savepath = "test_output/motion_sequence"
    save_frames(env.controller, actions, savepath,
                step_cb=step_cb, step_cb_args={"env": env})

if __name__ == "__main__":
    test_motion_sequence()
