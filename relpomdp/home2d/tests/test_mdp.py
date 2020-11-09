import pomdp_py
from sciex import Event
from relpomdp.home2d.tests.test_fake_slam import wait_for_action
from relpomdp.home2d.agent import NKAgent, FakeSLAM, Laser2DSensor, NKAgentViz
from relpomdp.home2d.domain.maps.build_map import random_world
from relpomdp.home2d.agent.transition_model import CanDeclareFound, DeclareFoundEffect
from relpomdp.home2d.domain.env import Home2DEnvironment
from relpomdp.home2d.agent.transition_model import DeclareFound
from relpomdp.home2d.constants import FILE_PATHS
from test_utils import add_target, random_policy_model, make_world
import copy

def build_mdp_agent(env, target_class,
                    target_sensor_config={}):
    """Build MDP agent"""
    robot_id = env.robot_id
    init_robot_pose = env.robot_state["pose"]
    nk_agent = NKAgent(robot_id, init_robot_pose, grid_map=env.grid_map)
    target_id = list(env.ids_for(target_class))[0]
    init_belief = pomdp_py.Histogram({env.state.object_states[target_id]:1.0})
    add_target(nk_agent, target_id, init_belief, env)
    sensor = Laser2DSensor(robot_id,
                           fov=target_sensor_config.get("fov", 90),
                           min_range=target_sensor_config.get("min_range", 1),
                           max_range=target_sensor_config.get("max_range", 2),
                           angle_increment=target_sensor_config.get("angle_increment", 0.1))
    nk_agent.add_sensor(sensor,
                        {target_class: target_sensor_config.get("noises", (0.99, 0.01))})
    return nk_agent


def step_mdp(env, agent, planner):
    """Runs a step in the MDP simulation"""
    # Plan action
    action = planner.plan(agent)

    # environment transitions and obtains reward (note that we use agent's reward model for convenience)
    env_state = env.state.copy()
    _ = env.state_transition(action, execute=True)
    env_next_state = env.state.copy()
    reward = agent.reward_model.sample(env_state, action, env_next_state)

    observation = agent.observation_model.sample(env.state, action)
    # update belief (only need to to so for robot belief)
    agent.belief.object_beliefs[env.robot_id] = pomdp_py.Histogram({
        env.robot_state.copy() : 1.0
    })
    planner.update(agent, action, observation)
    return action, copy.deepcopy(env.state), observation, reward


def test_mdp(env, target_class,
             discount_factor=0.95, max_depth=15,
             num_sims=300, exploration_constant=200,
             nsteps=100,
             target_sensor_config={},
             visualize=True,
             logger=None):

    target_id = list(env.ids_for(target_class))[0]
    nk_agent = build_mdp_agent(env, target_class,
                               target_sensor_config=target_sensor_config)
    policy_model = random_policy_model(nk_agent)

    agent = nk_agent.instantiate(policy_model)

    planner = pomdp_py.POUCT(max_depth=max_depth,
                             discount_factor=discount_factor,
                             num_sims=num_sims,
                             exploration_const=exploration_constant,
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
    for i in range(nsteps):
        # Visualize
        if visualize:
            viz.on_loop()
            viz.on_render()

        # Take a step
        action, next_state, observation, reward =\
            step_mdp(env, agent, planner)

        # Info and logging
        _step_info = "Step {} : Action: {}    Reward: {}    RobotPose: {}   TargetFound: {}"\
            .format(i+1, action, reward,
                    next_state.object_states[env.robot_id]["pose"],
                    next_state.object_states[target_id].get("is_found", False))
        if logger is None:
            print(_step_info)
        else:
            logger(_step_info)

        _rewards.append(reward)
        _states.append(next_state)
        _history.append((action, observation))

        # Termination check
        if isinstance(action, DeclareFound):
            if logger is None:
                print("Done!")
            else:
                logger("Done!")
            break
    if visualize:
        viz.on_cleanup()
    return _rewards, _states, _history

if __name__ == "__main__":
    env = make_world(seed=130)
    test_mdp(env, target_class="Salt", num_sims=1000)
