# This POMDP begins with no map

import pomdp_py
from relpomdp.home2d.agent import NKAgentViz, Laser2DSensor, NKAgent, FakeSLAM
from relpomdp.home2d.domain.maps.build_map import random_world
from relpomdp.home2d.agent.policy_model import GreedyActionPrior
from relpomdp.home2d.agent.transition_model import DeclareFound
from relpomdp.home2d.constants import FILE_PATHS
from relpomdp.home2d.utils import save_images_and_compress
from relpomdp.oopomdp.framework import Objstate, OOState, OOBelief
from relpomdp.home2d.tests.test_utils import add_target, random_policy_model, make_world, update_map,\
    preferred_policy_model, belief_fit_map
import copy
import time
import subprocess

def build_pomdp_nk_agent(env, target_class,
                         target_sensor_config={},
                         slam_sensor_config={}):
    """Build POMDP agent with no map initially"""
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
    return nk_agent, fake_slam


def step_pomdp_nk(env, nk_agent, fake_slam, planner, target_id,
                  logger=None):
    """Run a step in the simulation of POMDP without map knowledge"""
    # Make new agent for planning which uses the updated map and policy model
    # maintained by the nk_agent
    policy_model = preferred_policy_model(nk_agent,
                                          GreedyActionPrior,
                                          ap_args=[target_id])
    agent = nk_agent.instantiate(policy_model)
    planner.set_rollout_policy(agent.policy_model)

    robot_id = env.robot_id

    # Plan action
    start_time = time.time()
    action = planner.plan(agent)
    if logger is None:
        print("-------POUCT (took %.4fs) -----" % (time.time() - start_time))
        planner.print_action_values()

    # Environment transitions and obtains reward (note that we use agent's reward model for convenience)
    env_state = env.state.copy()
    prev_robot_pose = agent.belief.mpe().object_states[robot_id]["pose"]
    _ = env.state_transition(action, execute=True)
    env_next_state = env.state.copy()
    reward = agent.reward_model.sample(env_state, action, env_next_state)

    observation = agent.observation_model.sample(env.state, action)
    if logger is not None:
        logger(observation)
    else:
        print(observation)

    # Updates
    ## update belief of robot
    new_robot_belief = pomdp_py.Histogram({env.robot_state.copy() : 1.0})
    robot_pose = new_robot_belief.mpe()["pose"]

    ## update map (fake slam)
    update_map(fake_slam, nk_agent, prev_robot_pose, robot_pose, env)

    target_belief = nk_agent.object_belief(target_id)
    assert target_belief == agent.belief.object_beliefs[target_id], "Target belief mismatch; Unexpectedf."

    ## Update belief based on map update/expansion
    target_hist = belief_fit_map(target_belief, nk_agent.grid_map,
                                 get_dict=True)

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
    planner.update(agent, action, observation)

    return action, copy.deepcopy(env.state), observation, reward


def test_pomdp_nk(env, target_class,
                  discount_factor=0.95, max_depth=20,
                  num_sims=600, exploration_constant=200,
                  nsteps=100, save=False,
                  target_sensor_config={},
                  slam_sensor_config={},
                  visualize=True,
                  logger=None):
    robot_id = env.robot_id
    target_id = list(env.ids_for(target_class))[0]

    nk_agent, fake_slam = build_pomdp_nk_agent(env, target_class,
                                               target_sensor_config=target_sensor_config,
                                               slam_sensor_config=slam_sensor_config)

    planner = pomdp_py.POUCT(max_depth=max_depth,
                             discount_factor=discount_factor,
                             num_sims=num_sims,
                             exploration_const=exploration_constant)

    # policy_model = random_policy_model(nk_agent)
    # Visualize and run
    if visualize:
        viz = NKAgentViz(nk_agent,
                         env,
                         {},
                         res=30,
                         controllable=True,
                         img_path=FILE_PATHS["object_imgs"])
        viz.on_init()
    init_state = copy.deepcopy(env.state)
    _rewards = []
    _states = [init_state]
    _history = []
    _disc_reward = 0.0
    _gamma = 1.0
    game_states = []
    for i in range(nsteps):
        # Visualize
        if visualize:
            viz.on_loop()
            img_agent, img_world, img = viz.on_render(OOBelief(nk_agent.object_beliefs))
            game_states.append(img)

        # Take a step
        action, next_state, observation, reward = \
            step_pomdp_nk(env, nk_agent, fake_slam, planner, target_id, logger=logger)

        # Info and logging
        _disc_reward += _gamma*reward
        _gamma *= discount_factor
        _step_info = "Step {} : Action: {}   NumSims: {}    Reward: {}    DiscCumReward: {:.4f}    RobotPose: {}   TargetFound: {}"\
            .format(i+1, action, planner.last_num_sims,
                    reward, _disc_reward,
                    next_state.object_states[env.robot_id]["pose"],
                    next_state.object_states[target_id].get("is_found", False))
        if logger is None:
            print(_step_info)
        else:
            logger(_step_info)

        _rewards.append(reward)
        _states.append(next_state)
        _history.append((action, observation, copy.deepcopy(nk_agent.object_beliefs)))

        # Termination check
        if isinstance(action, DeclareFound):
            if logger is None:
                print("Done!")
            else:
                logger("Done!")
            break
    if visualize:
        game_states.append(img_world)
        viz.on_cleanup()

        if save:
            print("Saving images...")
            dirp = "./demos/test_pomdp_nk"
            save_images_and_compress(game_states,
                                     dirp)
            subprocess.Popen(["nautilus", dirp])
    return _rewards, _states, _history

if __name__ == "__main__":
    # To test an ordinary run: set seed to be 100. init robot pose (0,0,0)
    # To test the 'unable to see wall next to robot' issue, set seed to 1459,
    #    set init robot pose (0,1,0). Try a few times because doorway may open up differently
    env = make_world(seed=158, worldsize=(6,6))
    test_pomdp_nk(env, target_class="Salt", save=False, nsteps=100)
