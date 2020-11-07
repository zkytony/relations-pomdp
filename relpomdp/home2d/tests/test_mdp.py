import pomdp_py
from relpomdp.home2d.tests.test_fake_slam import wait_for_action
from relpomdp.home2d.agent.nk_agent import NKAgent, FakeSLAM
from relpomdp.home2d.agent.sensor import Laser2DSensor
from relpomdp.home2d.agent.visual import NKAgentViz
from relpomdp.home2d.domain.maps.build_map import random_world
from relpomdp.home2d.agent.transition_model import CanDeclareFound, DeclareFoundEffect
from relpomdp.home2d.domain.env import Home2DEnvironment
from relpomdp.home2d.agent.transition_model import DeclareFound
from relpomdp.home2d.constants import FILE_PATHS
from test_utils import add_target, random_policy_model, make_world
import copy

def test_mdp(env, target_class,
             discount_factor=0.95, max_depth=15,
             num_sims=300, exploration_constant=100,
             nsteps=100,
             target_sensor_config={},
             slam_sensor_config={}):
    robot_id = env.robot_id
    init_robot_pose = env.robot_state["pose"]
    nk_agent = NKAgent(robot_id, init_robot_pose, grid_map=env.grid_map)
    fake_slam = FakeSLAM(Laser2DSensor(robot_id,
                                       fov=slam_sensor_config.get("fov", 90),
                                       min_range=slam_sensor_config.get("min_range", 1),
                                       max_range=slam_sensor_config.get("max_range", 3),
                                       angle_increment=slam_sensor_config.get("angle_increment", 0.1)))
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
    policy_model = random_policy_model(nk_agent)

    agent = nk_agent.instantiate(policy_model)

    planner = pomdp_py.POUCT(max_depth=max_depth,
                             discount_factor=discount_factor,
                             num_sims=num_sims,
                             exploration_const=exploration_constant,
                             rollout_policy=agent.policy_model)

    # Visualize and run
    viz = NKAgentViz(agent,
                     env,
                     {},
                     res=30,
                     controllable=True,
                     img_path=FILE_PATHS["object_imgs"])
    viz.on_init()
    rewards = []
    for i in range(nsteps):
        # Visualize
        viz.on_loop()
        viz.on_render()

        action = planner.plan(agent)

        # environment transitions and obtains reward (note that we use agent's reward model for convenience)
        env_state = env.state.copy()
        _ = env.state_transition(action, execute=True)
        env_next_state = env.state.copy()
        reward = agent.reward_model.sample(env_state, action, env_next_state)

        observation = agent.observation_model.sample(env.state, action)
        # update belief (only need to to so for robot belief)
        agent.belief.object_beliefs[robot_id] = pomdp_py.Histogram({
            env.robot_state.copy() : 1.0
        })
        planner.update(agent, action, observation)
        print(action, reward)
        rewards.append(reward)
        if isinstance(action, DeclareFound):
            print("Done.")
            break
    viz.on_cleanup()
    return rewards

if __name__ == "__main__":
    env = make_world()
    test_mdp(env, target_class="Salt")
