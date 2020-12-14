from navigation import *
from relpomdp.realistic.utils.ai2thor_utils import save_frames,\
    plot_reachable_grid, get_reachable_pos_set
from relpomdp.realistic.environment import ThorEnv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import random
import numpy as np
import sys
import os


def test_motion_sequence():
    """This will launch a thor environment and
    manually control a robot to move inside.
    Modify `seq` below for different motions.

    Top-down view and reachable pos + robot pos are saved.

    This is a demo, not a unit test."""

    savepath = "test_output/motion_sequence2"

    def step_cb(step, event, env=None, ax=None, fig=None):
        # After every step (int), plot the agent pose on top of the
        # grid map.
        assert env is not None
        assert ax is not None
        assert fig is not None
        plot_reachable_grid(env.controller, ax, agent_pose=env.agent_pose())
        fig.canvas.draw()
        fig.canvas.flush_events()
        fig.savefig(os.path.join(savepath, "plot-%d.png" % step))
        time.sleep(0.3)

    motions = build_motion_actions()
    motions = {m.name:m for m in motions}

    config = {
        "scene_name": "FloorPlan_Train1_3",
        "agent_mode": "default",
        "width": 400,
        "height": 400
    }
    env = ThorEnv(config)
    env.launch()
    env.controller.step(action="ToggleMapView")

    plt.ion()
    fig = plt.figure(figsize=(6,3))
    ax = fig.add_subplot(1, 1, 1)#, projection="3d")
    plt.show(block=False)

    seq = ["MoveBack"]*5
           # "RotateLeft",
           # "RotateLeft",
           # "RotateRight",
           # "RotateRight",
           # "MoveBack",
           # "RotateLeft",
           # "MoveAhead",
           # "MoveAhead"]
    actions = [motions[name].to_thor_action() for name in seq]
    save_frames(env.controller, actions, savepath,
                step_cb=step_cb, step_cb_args={"env": env, "ax": ax, "fig": fig})
    print("Pass.")


def test_transition_model(grid_size=0.25, degrees=30, scene_name="FloorPlan_Train1_1"):
    """Tests whether the transition model predicts THOR's motion model"""
    config = {
        "scene_name": scene_name,
        "agent_mode": "default",
        "width": 400,
        "height": 400
    }
    motions = build_motion_actions(grid_size=grid_size, degrees=degrees)
    motions = {m.name:m for m in motions}

    env = ThorEnv(config)
    env.launch()
    reachable_positions = get_reachable_pos_set(env.controller, use_2d=True)

    state = NavState(*env.agent_pose(use_2d=True))
    T = TransitionModel(grid_size=grid_size)

    nrounds = 100
    for i in range(nrounds):
        # Randomly choose a motion action. Execute it
        # in THOR, and then sample the next state.
        # assert that both get the same pose, if both are reachable.
        sys.stdout.write("[%d/%d]\r" % (i+1, nrounds))
        sys.stdout.flush()

        motion_action = motions[random.sample(list(motions), 1)[0]]
        thor_action_name, params = motion_action.to_thor_action()
        event = env.controller.step(action=thor_action_name, **params)

        next_state = T.sample(state, motion_action)
        predicted_pose = (next_state.pos, round(next_state.rot, 2) % 360.0)
        if next_state.pos not in reachable_positions:
            continue
        else:
            pos, rot = env.agent_pose(use_2d=True)
            expected_pose = (pos, round(rot, 2) % 360.0)
            assert predicted_pose == expected_pose,\
                "Expecting pose at {}, predicted {}. Previous pose {}, action {}."\
                .format(expected_pose, predicted_pose, (state.pos, state.rot), motion_action)
            state = next_state
    print("\nPass.")


def test_transition_model_forced(grid_size=0.25, degrees=30, scene_name="FloorPlan_Train1_1"):
    """Tests whether we can force THOR to use the transition model's output"""
    config = {
        "scene_name": scene_name,
        "agent_mode": "default",
        "width": 400,
        "height": 400
    }
    motions = build_motion_actions(grid_size=grid_size, degrees=degrees)
    motions = {m.name:m for m in motions}

    env = ThorEnv(config)
    env.launch()
    reachable_positions = get_reachable_pos_set(env.controller, use_2d=True)

    init_x, init_y, init_z = env.agent_pose()[0]
    state = NavState(*env.agent_pose(use_2d=True))
    T = TransitionModel(grid_size=grid_size)

    nrounds = 100
    for i in range(nrounds):
        # Randomly choose a motion action. Execute it
        # in THOR, and then sample the next state.
        # assert that both get the same pose, if both are reachable.
        sys.stdout.write("[%d/%d]\r" % (i+1, nrounds))
        sys.stdout.flush()

        motion_action = motions[random.sample(list(motions), 1)[0]]
        next_state = T.sample(state, motion_action)
        predicted_pose = (next_state.pos, round(next_state.rot, 2) % 360.0)
        if next_state.pos not in reachable_positions:
            continue
        else:
            # Force THOR to be at sampled pose
            # NOTE: because of ai2thor's BUG, you must specify y to be initial y.
            env.controller.step('TeleportFull',
                                x=next_state.pos[0], y=init_y, z=next_state.pos[1],
                                rotation=dict(y=next_state.rot))

            pos, rot = env.agent_pose(use_2d=True)
            expected_pose = (pos, round(rot, 2) % 360.0)
            assert predicted_pose == expected_pose,\
                "Expecting pose at {}, predicted {}. Previous pose {}, action {}."\
                .format(expected_pose, predicted_pose, (state.pos, state.rot), motion_action)
            state = next_state
    print("\nPass.")


if __name__ == "__main__":
    # test_motion_sequence()
    test_transition_model(grid_size=0.25, degrees=90, scene_name="FloorPlan_Train1_1")
    test_transition_model_forced(grid_size=0.25, degrees=90, scene_name="FloorPlan_Train1_3")
    test_transition_model_forced(grid_size=0.25, degrees=90, scene_name="FloorPlan_Train1_5")
    test_transition_model_forced(grid_size=0.25, degrees=30, scene_name="FloorPlan_Train1_2")
    test_transition_model_forced(grid_size=0.25, degrees=45, scene_name="FloorPlan_Train1_4")
