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
from relpomdp.home2d.utils import save_images_and_compress
from relpomdp.oopomdp.framework import Objstate, OOState
from test_utils import add_pickup_target, random_policy_model, make_world, update_map
import copy
import subprocess


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
    add_pickup_target(nk_agent, target_id, init_belief, env)
    sensor = Laser2DSensor(robot_id,
                           fov=90, min_range=1,
                           max_range=2, angle_increment=0.1)
    nk_agent.add_sensor(sensor, {target_class: (1000., 0.1)})
    policy_model = random_policy_model(nk_agent)

    agent = nk_agent.instantiate(policy_model)

    planner = pomdp_py.POUCT(max_depth=15,
                             discount_factor=discount_factor,
                             num_sims=300,
                             exploration_const=100,
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
        print("-------POUCT-----")
        planner.print_action_values()
        print("-----------------")

        # environment transitions and obtains reward (note that we use agent's reward model for convenience)
        env_state = env.state.copy()
        prev_robot_pose = agent.belief.mpe().object_states[robot_id]["pose"]
        _ = env.state_transition(action, execute=True)
        env_next_state = env.state.copy()
        reward = agent.reward_model.sample(env_state, action, env_next_state)

        observation = agent.observation_model.sample(env.state, action)

        # update belief of robot
        new_robot_belief = pomdp_py.Histogram({env.robot_state.copy() : 1.0})
        robot_pose = new_robot_belief.mpe()["pose"]

        # update map (fake slam)
        update_map(fake_slam, nk_agent, prev_robot_pose, robot_pose, env)
        partial_map = nk_agent.grid_map

        # Belief update.
        ## First obtain the current belief, taking into account the frontier
        target_hist = {}
        target_belief = nk_agent.object_belief(target_id)
        assert target_belief == agent.belief.object_beliefs[target_id], "Target belief mismatch; Unexpected."
        for x, y in partial_map.frontier() | partial_map.free_locations:
            target_state = Objstate(target_class, pose=(x,y))
            if target_state in target_belief:
                target_hist[target_state] = target_belief[target_state]
            else:
                # Assign a uniform belief to the frontier; 1.0 / len(target_belief)
                # means that the belief at frontier will be no higher or lower than
                # the belief at any location currently in the explored map, by default
                # (which is uniform 1.0 / len(target_belief))
                target_hist[target_state] = 1.0 / len(target_belief)
        ## Then, renormalize
        prob_sum = sum(target_hist[state] for state in target_hist)
        for target_state in target_hist:
            target_hist[target_state] /= prob_sum

        ## Now, do belief update based on observation
        next_target_hist = {}
        total_prob = 0.0
        for target_state in target_hist:
            robot_state = new_robot_belief.mpe()
            oostate = OOState({robot_id: robot_state,
                               target_id: target_state})
            obs_prob = agent.observation_model.probability(observation, oostate, action)
            next_target_hist[target_state] = obs_prob * target_hist[target_state]
            total_prob += next_target_hist[target_state]
        for target_state in next_target_hist:
            next_target_hist[target_state] /= total_prob

        # Finally, update the agent
        ## Update object belief
        nk_agent.set_belief(robot_id, new_robot_belief)
        nk_agent.set_belief(target_id, pomdp_py.Histogram(next_target_hist))
        ## Generate policy model
        policy_model = random_policy_model(nk_agent, memory=agent.policy_model.memory)
        ## Make new agent which uses the new policy mode made on top of new map
        agent = nk_agent.instantiate(policy_model)
        planner.set_rollout_policy(agent.policy_model)


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
    test_pomdp_nk(env, save=False, nsteps=30)
