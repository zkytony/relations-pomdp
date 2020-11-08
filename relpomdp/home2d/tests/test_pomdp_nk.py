# This POMDP begins with no map

import pomdp_py
from relpomdp.home2d.tests.test_fake_slam import wait_for_action
from relpomdp.home2d.agent import NKAgentViz, Laser2DSensor, NKAgent, FakeSLAM
from relpomdp.home2d.agent.transition_model import CanDeclareFound, DeclareFoundEffect
from relpomdp.home2d.domain.maps.build_map import random_world
from relpomdp.home2d.agent.policy_model import GreedyActionPrior
from relpomdp.home2d.domain.action import MoveN
from relpomdp.home2d.domain.env import Home2DEnvironment
from relpomdp.home2d.agent.transition_model import DeclareFound
from relpomdp.home2d.constants import FILE_PATHS
from relpomdp.home2d.utils import save_images_and_compress
from relpomdp.oopomdp.framework import Objstate, OOState
from test_utils import add_target, random_policy_model, make_world, update_map,\
    preferred_policy_model
import copy
import time
import subprocess


def test_pomdp_nk(env, target_class,
                  discount_factor=0.95, max_depth=20,
                  num_sims=600, exploration_constant=200,
                  nsteps=100, save=False,
                  target_sensor_config={},
                  slam_sensor_config={}):
    robot_id = env.robot_id
    init_robot_pose = env.robot_state["pose"]
    nk_agent = NKAgent(robot_id, init_robot_pose)
    fake_slam = FakeSLAM(Laser2DSensor(robot_id,
                                       fov=slam_sensor_config.get("fov", 90),
                                       min_range=slam_sensor_config.get("min_range", 1),
                                       max_range=slam_sensor_config.get("max_range", 3),
                                       angle_increment=slam_sensor_config.get("angle_increment", 0.1)))
    target_id = list(env.ids_for(target_class))[0]

    # Uniform belief over free spaces and a layer of frontier
    frontier = nk_agent.grid_map.frontier()
    print(frontier)

    target_hist = {}
    total_prob = 0.0
    for x, y in nk_agent.grid_map.free_locations | frontier:
        target_state = Objstate(target_class, pose=(x,y))
        if (x,y) == init_robot_pose[:2]:
            # the target won't be there
            target_hist[target_state] = 1e-12
        else:
            target_hist[target_state] = 1.
        total_prob += target_hist[target_state]
    for state in target_hist:
        target_hist[state] /= total_prob

    init_belief = pomdp_py.Histogram(target_hist)
    add_target(nk_agent, target_id, init_belief, env)
    sensor = Laser2DSensor(robot_id,
                           fov=target_sensor_config.get("fov", 90),
                           min_range=target_sensor_config.get("min_range", 1),
                           max_range=target_sensor_config.get("max_range", 2),
                           angle_increment=target_sensor_config.get("angle_increment", 0.1))
    nk_agent.add_sensor(sensor,
                        {target_class: target_sensor_config.get("noises", (0.99, 0.01))})
    # policy_model = random_policy_model(nk_agent)
    policy_model = preferred_policy_model(nk_agent,
                                          GreedyActionPrior,
                                          ap_args=[target_id])

    agent = nk_agent.instantiate(policy_model)

    planner = pomdp_py.POUCT(max_depth=max_depth,
                             discount_factor=discount_factor,
                             num_sims=num_sims,
                             exploration_const=exploration_constant,
                             rollout_policy=agent.policy_model)

    # Visualize and run
    viz = NKAgentViz(nk_agent,
                     env,
                     {},
                     res=30,
                     controllable=True,
                     img_path=FILE_PATHS["object_imgs"])
    viz.on_init()
    rewards = []
    game_states = []
    for i in range(nsteps):
        # Visualize
        viz.on_loop()
        img, img_world = viz.on_render(agent.belief)

        start_time = time.time()
        action = planner.plan(agent)
        print("-------POUCT (took %.4fs) -----" % (time.time() - start_time))
        planner.print_action_values()
        print("-----------------")

        # Environment transitions and obtains reward (note that we use agent's reward model for convenience)
        env_state = env.state.copy()
        prev_robot_pose = agent.belief.mpe().object_states[robot_id]["pose"]
        _ = env.state_transition(action, execute=True)
        env_next_state = env.state.copy()
        reward = agent.reward_model.sample(env_state, action, env_next_state)

        observation = agent.observation_model.sample(env.state, action)
        print(observation)

        # Updates
        ## update belief of robot
        new_robot_belief = pomdp_py.Histogram({env.robot_state.copy() : 1.0})
        robot_pose = new_robot_belief.mpe()["pose"]

        ## update map (fake slam)
        update_map(fake_slam, nk_agent, prev_robot_pose, robot_pose, env)
        partial_map = nk_agent.grid_map
        updated_map_locations = partial_map.frontier() | partial_map.free_locations

        ## Belief update.
        target_belief = nk_agent.object_belief(target_id)
        assert target_belief == agent.belief.object_beliefs[target_id], "Target belief mismatch; Unexpectedf."
        ### Belief at state B(s) = Val(s) / Norm, where Val is the unnormalized belief,
        ### and Norm is some normalizer. Here, we will regard the number of grid cells
        ### in a map as the normalizer, and compute the unnormalized belief accordingly.
        ### Basically we want to rescale the normalized belief to fit onto the updated map.
        cur_norm = len(target_belief)
        new_norm = len(updated_map_locations)

        new_norm_target_hist = {state:target_belief[state]*(cur_norm/new_norm) for state in target_belief}
        updated_total_prob = 1. - sum(new_norm_target_hist.values()) # The total unnormalized probability in the expanded region

        target_hist = {}
        for x, y in updated_map_locations:
            target_state = Objstate(target_class, pose=(x,y))
            if target_state in new_norm_target_hist:
                target_hist[target_state] = new_norm_target_hist[target_state]
            else:
                if new_norm < cur_norm:
                    # Not going to track belief for this state. This state
                    # should lie outside of the map boundary
                    assert not (0 <= x < env.grid_map.width)\
                        or not (0 <= y < env.grid_map.length)
                    continue

                if new_norm - cur_norm == 0:
                    # The map did not expand, but we encounter a new target state.
                    # This can happen when target state is outside of the boundary wall
                    assert abs(updated_total_prob) <= 1e-9
                    target_hist[target_state] = updated_total_prob
                else:
                    target_hist[target_state] = updated_total_prob / (new_norm - cur_norm)
        ## Then, renormalize
        prob_sum = sum(target_hist[state] for state in target_hist)

        for target_state in target_hist:
            assert target_hist[target_state] >= -1e-9,\
                "Belief {} is invalid".format(target_hist[target_state])
            target_hist[target_state] = max(target_hist[target_state], 1e-32)
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
        # policy_model = random_policy_model(nk_agent, memory=agent.policy_model.memory)
        planner.update(agent, action, observation)
        tree = agent.tree
        policy_model = preferred_policy_model(nk_agent,
                                              GreedyActionPrior,
                                              ap_args=[target_id])
        ## Make new agent which uses the new policy mode made on top of new map
        agent = nk_agent.instantiate(policy_model)
        planner.set_rollout_policy(agent.policy_model)

        print(action, reward)
        rewards.append(reward)
        game_states.append(img)
        if isinstance(action, DeclareFound):
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
    # To test an ordinary run: set seed to be 100. init robot pose (0,0,0)
    # To test the 'unable to see wall next to robot' issue, set seed to 1459,
    #    set init robot pose (0,1,0). Try a few times because doorway may open up differently
    env = make_world(seed=1459, worldsize=(10,10))
    test_pomdp_nk(env, target_class="Salt", save=False, nsteps=100)
