# This POMDP begins with no map

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
    init_state, grid_map = random_world(10, 10, 3,
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


def test_pomdp_nk():
    env = make_world()
    robot_id = env.robot_id
    init_robot_pose = env.robot_state["pose"]
    nk_agent = NKAgent(robot_id, init_robot_pose)
    fake_slam = FakeSLAM(Laser2DSensor(robot_id,
                                       fov=90, min_range=1,
                                       max_range=5, angle_increment=0.1))

    target_class = "Salt"
    target_id = list(env.ids_for(target_class))[0]

    # Uniform belief over free spaces and a layer of frontier
    frontier = nk_agent.grid_map.frontier()
    print(frontier)

    target_hist = {}
    total_prob = 0.0
    for x, y in nk_agent.grid_map.free_locations | frontier:
        # if (x,y) == init_robot_pose[:2]:
        #     continue  # skip the robot's own pose because the target won't be there
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
    viz = NKAgentViz(nk_agent,
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

        # update map (fake slam)
        robot_pose = agent.belief.object_beliefs[robot_id].mpe()["pose"]
        fake_slam.update(nk_agent.grid_map, robot_pose, env)
        nk_agent.update()  # Update the nk_agent because policy model needs to be updated
        tree = agent.tree
        agent = nk_agent.instantiate(agent.belief)  # TODO: REFACTOR; pomdp_py doesn't allow reassigning models to agents
        planner.set_rollout_policy(agent.policy_model)
        agent.tree = tree
        # agent.grid_map = nk_agent.grid_map  # make sure the planning agent's map is updated

        # Belief update.
        ## First obtain the current belief
        target_belief = agent.belief.object_beliefs[target_id]
        cur_locs = set(s["pose"] for s in target_belief)  # obtain current locations modeled by the belief
        avg_belief = 1.0 / len(target_belief)

        ## Then, assign a uniform belief (with custom initial value) to
        ## locations that are newly added (i.e. new frontier and free locs)
        frontier = agent.grid_map.frontier()
        free_locs = agent.grid_map.free_locations
        target_hist = {}
        for x, y in (free_locs | frontier):
            target_state = Objstate(target_class, pose=(x,y))
            if (x,y) in cur_locs:
                target_hist[target_state] = target_belief[target_state]
            else:
                target_hist[target_state] = avg_belief

        ## Then, renormalize
        prob_sum = sum(target_hist[state] for state in target_hist)
        for target_state in target_hist:
            target_hist[target_state] /= prob_sum

        ## Now, do belief update based on observation
        next_target_hist = {}
        total_prob = 0.0
        for target_state in target_hist:
            robot_state = agent.belief.object_beliefs[robot_id].mpe()
            oostate = OOState({robot_id: robot_state,
                               target_id: target_state})
            obs_prob = agent.observation_model.probability(observation, oostate, action)
            next_target_hist[target_state] = obs_prob * target_hist[target_state]
            total_prob += next_target_hist[target_state]
        for target_state in next_target_hist:
            next_target_hist[target_state] /= total_prob
        agent.belief.object_beliefs[target_id] = pomdp_py.Histogram(next_target_hist)
        agent.belief.object_beliefs[target_id] = pomdp_py.Histogram(next_target_hist)
        planner.update(agent, action, observation)
        print(action, reward)
        if isinstance(action, Pickup):
            print("Done.")
            break

if __name__ == "__main__":
    test_pomdp_nk()
