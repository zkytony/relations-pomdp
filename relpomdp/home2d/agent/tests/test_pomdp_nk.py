# This POMDP begins with no map

import pomdp_py
from relpomdp.home2d.agent.tests.test_fake_slam import wait_for_action
from relpomdp.home2d.agent.nk_agent import NKAgent, FakeSLAM
from relpomdp.home2d.tasks.common.sensor import Laser2DSensor
from relpomdp.home2d.agent.visual import NKAgentViz
from relpomdp.home2d.agent.transition_model import CanPickup, PickupEffect
from relpomdp.home2d.domain.maps.build_map import random_world
from relpomdp.home2d.domain.action import MoveN
from relpomdp.home2d.domain.env import Home2DEnvironment
from relpomdp.home2d.agent.transition_model import Pickup
from relpomdp.oopomdp.framework import Objstate, OOState
from relpomdp.home2d.utils import save_images_and_compress
import copy
import subprocess

def make_world():
    robot_id = 0
    init_robot_pose = (0, 0, 0)
    init_state, grid_map = random_world(6, 6, 3,
                                        ["Kitchen", "Office", "Office"],
                                        objects={"Office": {"Computer": (1, (1,1))},
                                                 "Kitchen": {"Pepper": (1, (1,1)),
                                                             "Salt": (1, (1,1))},
                                                 "Bathroom": {"Toilet": (1, (1,1))}},
                                        robot_id=robot_id, init_robot_pose=init_robot_pose,
                                        ndoors=1,
                                        seed=10)
    env = Home2DEnvironment(robot_id,
                            grid_map,
                            init_state)
    return env

def env_add_target(env, target_id, target_class):
    """
    Adds target to search for (adds pickup action and effect)
    """


def test_pomdp_nk(env, nsteps=100, discount_factor=0.95, save=False):
    robot_id = env.robot_id
    init_robot_pose = env.robot_state["pose"]
    nk_agent = NKAgent(robot_id, init_robot_pose)
    fake_slam = FakeSLAM(Laser2DSensor(robot_id,
                                       fov=90, min_range=1,
                                       max_range=3, angle_increment=0.1))

    target_class = "Salt"
    target_id = list(env.ids_for(target_class))[0]

    # Uniform belief over free spaces and a layer of frontier
    frontier = nk_agent.grid_map.frontier()
    print(frontier)

    target_hist = {}
    total_prob = 0.0
    for x, y in nk_agent.grid_map.free_locations | frontier:
        if (x,y) == init_robot_pose[:2]:
            continue  # skip the robot's own pose because the target won't be there
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

    # Need to update R/T models of the environment for the search task
    env.set_reward_model(agent.reward_model)
    pickup_condeff = (CanPickup(env.robot_id, target_id), PickupEffect())
    env.transition_model.cond_effects.append(pickup_condeff)

    planner = pomdp_py.POUCT(max_depth=20,
                             discount_factor=discount_factor,
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
    rewards = []
    game_states = []
    for i in range(nsteps):
        # Visualize
        viz.on_loop()
        img, img_world = viz.on_render(agent.belief)

        action = planner.plan(agent)

        # environment transitions and obtains reward (note that we use agent's reward model for convenience)
        env_state = env.state.copy()
        prev_robot_pose = agent.belief.mpe().object_states[robot_id]["pose"]
        _ = env.state_transition(action, execute=True)
        env_next_state = env.state.copy()
        reward = agent.reward_model.sample(env_state, action, env_next_state)

        observation = agent.observation_model.sample(env.state, action)
        # update belief of robot
        agent.belief.object_beliefs[robot_id] = pomdp_py.Histogram({
            env.robot_state.copy() : 1.0
        })
        robot_pose = agent.belief.mpe().object_states[robot_id]["pose"]

        # update map (fake slam)
        fake_slam.update(nk_agent.grid_map, prev_robot_pose, robot_pose, env)
        # Update the nk_agent because policy model needs to be updated, because
        # its grid_map was just updated
        nk_agent.update(robot_pose, prev_robot_pose, action)
        # tree = agent.tree
        agent = nk_agent.instantiate(agent.belief)  # TODO: REFACTOR; pomdp_py doesn't allow reassigning models to agents
        planner.set_rollout_policy(agent.policy_model)
        # agent.tree = tree

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
        # planner update is futile because the map changes
        # planner.update(agent, action, observation)
        print(action, reward)
        rewards.append(reward)
        game_states.append(img)
        if isinstance(action, Pickup):
            print("Done.")
            break
    game_states.append(img_world)
    viz.on_cleanup()

    if save:
        print("Saving images...")
        dirp = "./demos/test_pomdp_nk"
        save_images_and_compress(game_states,
                                 dirp)
        subprocess.Popen(["nautilus", dirp])
    return rewards

if __name__ == "__main__":
    env = make_world()
    test_pomdp_nk(env, save=True, nsteps=30)
