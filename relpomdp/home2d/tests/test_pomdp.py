# This POMDP has access to the full map
# Begins with uniform prior

import pomdp_py
from relpomdp.home2d.agent import NKAgent, FakeSLAM, Laser2DSensor, NKAgentViz
from relpomdp.home2d.agent.policy_model import GreedyActionPrior
from relpomdp.home2d.agent.transition_model import DeclareFound
from relpomdp.home2d.constants import FILE_PATHS
from relpomdp.oopomdp.framework import Objstate, OOState
from relpomdp.home2d.utils import save_images_and_compress
from relpomdp.home2d.tests.test_utils import add_target, random_policy_model, make_world,\
    preferred_policy_model
import subprocess
import copy

def build_pomdp_agent(env, target_class,
                      target_sensor_config={},
                      slam_sensor_config={}):
    """Build POMDP agent"""
    robot_id = env.robot_id
    init_robot_pose = env.robot_state["pose"]
    # The agent can access the full map
    nk_agent = NKAgent(robot_id, init_robot_pose, grid_map=env.grid_map)
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
    add_target(nk_agent, target_id, init_belief, env)
    sensor = Laser2DSensor(robot_id,
                           fov=target_sensor_config.get("fov", 90),
                           min_range=target_sensor_config.get("min_range", 1),
                           max_range=target_sensor_config.get("max_range", 2),
                           angle_increment=target_sensor_config.get("angle_increment", 0.1))
    nk_agent.add_sensor(sensor,
                        {target_class: target_sensor_config.get("noises", (0.8, 0.01))})
    return nk_agent


def step_pomdp(env, agent, planner, target_id, logger=None):
    """Runs a step in the POMDP simulation"""
    robot_id = env.robot_id

    # Plan action
    action = planner.plan(agent)
    if logger is None:
        print("------POUCT------")
        planner.print_action_values()

    # environment transitions and obtains reward (note that we use agent's reward model for convenience)
    env_state = env.state.copy()
    _ = env.state_transition(action, execute=True)
    env_next_state = env.state.copy()
    reward = agent.reward_model.sample(env_state, action, env_next_state)

    observation = agent.observation_model.sample(env.state, action)
    if logger is not None:
        logger(observation)
    else:
        print(observation)

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
    return action, copy.deepcopy(env.state), observation, reward


def test_pomdp(env, target_class,
               discount_factor=0.95, max_depth=15,
               num_sims=300, exploration_constant=200,
               nsteps=100, save=False,
               target_sensor_config={},
               visualize=True,
               logger=None):
    robot_id = env.robot_id
    target_id = list(env.ids_for(target_class))[0]

    nk_agent = build_pomdp_agent(env, target_class,
                                 target_sensor_config=target_sensor_config)

    policy_model = preferred_policy_model(nk_agent,
                                          GreedyActionPrior,
                                          ap_args=[target_id])

    agent = nk_agent.instantiate(policy_model)

    planner = pomdp_py.POUCT(max_depth=max_depth,
                             discount_factor=discount_factor,
                             num_sims=num_sims,
                             exploration_const=exploration_constant,  # setting this to 200 makes the agent hesitate in where it is
                             rollout_policy=agent.policy_model)

    # Visualize and run
    if visualize:
        viz = NKAgentViz(agent,
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
    game_imgs = []
    for i in range(nsteps):
        # Visualize
        if visualize:
            viz.on_loop()
            img, img_world = viz.on_render(agent.belief)
            game_imgs.append(img)

        # Take a step
        action, next_state, observation, reward =\
            step_pomdp(env, agent, planner, target_id, logger=logger)

        # Info and logging
        _disc_reward += _gamma*reward
        _gamma *= discount_factor
        _step_info = "Step {} : Action: {}   NumSims,  Reward: {}    DiscCumReward: {:.4f}    RobotPose: {}   TargetFound: {}"\
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
        _history.append((action, observation, copy.deepcopy(agent.belief)))

        # Termination check
        if isinstance(action, DeclareFound):
            if logger is None:
                print("Done!")
            else:
                logger("Done!")
            break
    if visualize:
        viz.on_cleanup()
        game_imgs.append(img_world)

        if save:
            print("Saving images...")
            dirp = "./demos/test_pomdp"
            save_images_and_compress(game_imgs,
                                     dirp)
            subprocess.Popen(["nautilus", dirp])

    return _rewards, _states, _history

if __name__ == "__main__":
    env = make_world()
    test_pomdp(copy.deepcopy(env), target_class="Salt",
               save=False, nsteps=50)
