# Tests for object search

from object_search import *
import sys


# Test policy model
def test_policy_model(scene_name, grid_size=0.25, degrees=90):
    config = {
        "scene_name": scene_name,
        "agent_mode": "default",
        "width": 400,
        "height": 400,
        "grid_size": grid_size
    }
    motions = build_motion_actions(grid_size=grid_size, degrees=degrees)
    motions_dict = {m.name:m for m in motions}

    env = ThorEnv(config)
    env.launch()
    reachable_positions = get_reachable_pos_set(env.controller, use_2d=True)

    policy_model = PolicyModel(motions | {DeclareFound()},
                               reachable_positions, grid_size=grid_size)
    transition_model = TransitionModel(grid_size=grid_size)

    nrounds = 200
    for i in range(nrounds):
        sys.stdout.write("[%d/%d]\r" % (i+1, nrounds))
        sys.stdout.flush()

        robot_pose = (random.sample(reachable_positions, 1)[0],
                      random.uniform(0, 360))
        state = State(RobotState(robot_pose, False),
                      TargetState("Box", random.sample(reachable_positions, 1)[0]))

        for name in {"MoveAhead", "MoveBack"}:
            action = motions_dict[name]
            next_state = transition_model.sample(state, action)
            if next_state.robot_pose[0] in reachable_positions:
                assert action in policy_model.get_all_actions(state=state)
            else:
                assert action not in policy_model.get_all_actions(state=state)
    print("\nPassed.")



if __name__ == "__main__":
    # test_system("FloorPlan_Train1_1", grid_size=0.5)
    test_policy_model("FloorPlan_Train1_1", grid_size=0.5)
