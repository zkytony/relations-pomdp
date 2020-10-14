# This POMDP has access to the full map
# Begins with uniform prior

import pomdp_py
from relpomdp.home2d.agent.tests.test import wait_for_action
from relpomdp.home2d.agent.nk_agent import NKAgent, FakeSLAM
from relpomdp.home2d.tasks.common.sensor import Laser2DSensor
from relpomdp.home2d.agent.visual import NKAgentViz
from relpomdp.home2d.domain.maps.build_map import random_world
from relpomdp.home2d.domain.env import Home2DEnvironment
from relpomdp.home2d.agent.transition_model import Pickup
from relpomdp.oopomdp.framework import Objstate, OOState
import copy

def make_world():
    robot_id = 0
    init_robot_pose = (0, 0, 0)
    init_state, grid_map = random_world(6, 6, 3,
                                        ["Kitchen", "Office", "Office"],
                                        objects={"Office": {"Computer": (1, (1,1))},
                                                 "Kitchen": {"Pepper": (1, (1,1))},
                                                 "Bathroom": {"Toilet": (1, (1,1))},
                                                 "Corridor": {"Salt": (1, (1,1))}},
                                        robot_id=robot_id, init_robot_pose=init_robot_pose)
    env = Home2DEnvironment(robot_id,
                            grid_map,
                            init_state)
    return env


def test_pomdp():
    env = make_world()
    robot_id = env.robot_id
    init_robot_pose = env.robot_state["pose"]
    # The agent can access the full map
    nk_agent = NKAgent(robot_id, init_robot_pose, grid_map=env.grid_map)
    fake_slam = FakeSLAM(Laser2DSensor(robot_id,
                                       fov=90, min_range=1,
                                       max_range=3, angle_increment=0.1))

    target_class = "Salt"
    target_id = list(env.ids_for(target_class))[0]

    # Uniform belief
    target_hist = {}
    total_prob = 0.0
    for x in range(env.grid_map.width):
        for y in range(env.grid_map.width):
            target_state = Objstate(target_class, pose=(x,y))
            target_hist[target_state] = 1.
            total_prob += target_hist[target_state]
    for state in target_hist:
        target_hist[state] /= total_prob

    init_belief = pomdp_py.Histogram(target_hist)
    nk_agent.add_target(target_id, target_class, init_belief)
    sensor = Laser2DSensor(robot_id,
                           fov=90, min_range=1,
                           max_range=2, angle_increment=0.1)
    nk_agent.add_sensor(sensor, {target_class: (100., 0.1)})
    nk_agent.update()

    agent = nk_agent.instantiate()
    planner = pomdp_py.POUCT(max_depth=20,
                             discount_factor=0.95,
                             num_sims=1000,
                             exploration_const=200,
                             rollout_policy=agent.policy_model)

    # Visualize and run
    viz = NKAgentViz(agent,
                     env,
                     {},
                     res=30,
                     controllable=True,
                     img_path="../../domain/imgs")
    viz.on_init()
    for i in range(100):
        # Visualize
        viz.on_loop()
        viz.on_render(agent.belief)

        action = planner.plan(agent)

        # environment transitions and obtains reward (note that we use agent's reward model for convenience)
        env_state = env.state.copy()
        _ = env.state_transition(action, execute=True)
        env_next_state = env.state.copy()
        reward = agent.reward_model.sample(env_state, action, env_next_state)

        observation = agent.observation_model.sample(env.state, action)
        # update belief of robot
        agent.belief.object_beliefs[robot_id] = pomdp_py.Histogram({
            env.robot_state.copy() : 1.0
        })

        # update belief of target
        next_target_hist = {}
        target_belief = agent.belief.object_beliefs[target_id]
        total_prob = 0.0
        for target_state in target_belief:
            robot_state = agent.belief.object_beliefs[robot_id].mpe()
            oostate = OOState({robot_id: robot_state,
                               target_id: target_state})
            obs_prob = agent.observation_model.probability(observation, oostate, action)
            next_target_hist[target_state] = obs_prob * target_belief[target_state]
            total_prob += next_target_hist[target_state]
        for target_state in next_target_hist:
            next_target_hist[target_state] /= total_prob
        agent.belief.object_beliefs[target_id] = pomdp_py.Histogram(next_target_hist)
        planner.update(agent, action, observation)
        print(action, reward)
        if isinstance(action, Pickup):
            print("Done.")
            break

if __name__ == "__main__":
    test_pomdp()
