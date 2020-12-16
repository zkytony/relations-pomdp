# Tests for object search

from object_search import *
import sys
import time


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


def test_sensor_model(scene_name, grid_size=0.25, degrees=90):
    # Test the field of view
    world_size = 20
    sensor = FanSensor(fov=90, min_range=0.0, max_range=grid_size*2)
    sensor_model = SensorObservationModel(sensor, "Box", 0.9)

    plt.ion()
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(1, 1, 1)#, projection="3d")
    ax.set_aspect("equal")
    plt.show(block=False)

    x_coords = [i*grid_size for i in range(world_size)]
    z_coords = [i*grid_size for i in range(world_size)]
    z, x = np.meshgrid(z_coords, x_coords)
    ax.scatter(x.flatten(), z.flatten(), s=10.0, c='r')
    positions = list(zip(x.flatten(),z.flatten()))

    robot_pose = ((0.0, 0.0), 0.0)
    plot_robot(ax, robot_pose)
    sensor_model.visualize(ax, robot_pose, positions)

    robot_pose = ((10.0*grid_size, 10.0*grid_size), 0.0)
    plot_robot(ax, robot_pose)
    sensor_model.visualize(ax, robot_pose, positions)

    robot_pose = ((12.0*grid_size, 5.0*grid_size), 45.0)
    plot_robot(ax, robot_pose)
    sensor_model.visualize(ax, robot_pose, positions)

    robot_pose = ((1.0*grid_size, 10.0*grid_size), 30.0)
    plot_robot(ax, robot_pose)
    sensor_model.visualize(ax, robot_pose, positions)

    robot_pose = ((7.0*grid_size, 7.0*grid_size), 180.0)
    plot_robot(ax, robot_pose, color='g')
    sensor_model.visualize(ax, robot_pose, positions)

    robot_pose = ((15.0*grid_size, 15.0*grid_size), 270.0)
    plot_robot(ax, robot_pose, color='g')
    sensor_model.visualize(ax, robot_pose, positions)

    robot_pose = ((18.0*grid_size, 5.0*grid_size), 225.0)
    plot_robot(ax, robot_pose, color='g')
    sensor_model.visualize(ax, robot_pose, positions)

    robot_pose = ((11.0*grid_size, 2.0*grid_size), 130.0)
    plot_robot(ax, robot_pose)
    sensor_model.visualize(ax, robot_pose, positions)

    ax.set_xlabel("X")
    ax.set_ylabel("Z")

    fig.canvas.draw()
    fig.canvas.flush_events()
    time.sleep(5)
    print("Pass.")


if __name__ == "__main__":
    # test_system("FloorPlan_Train1_1", grid_size=0.5)
    # test_policy_model("FloorPlan_Train1_1", grid_size=0.5)
    test_sensor_model("FloorPlan30", grid_size=0.25, degrees=90)
