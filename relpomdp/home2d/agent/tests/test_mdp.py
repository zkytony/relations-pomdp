import pomdp_py
from relpomdp.home2d.agent.tests.test_fake_slam import wait_for_action
from relpomdp.home2d.agent.nk_agent import NKAgent, FakeSLAM
from relpomdp.home2d.tasks.common.sensor import Laser2DSensor
from relpomdp.home2d.agent.visual import NKAgentViz
from relpomdp.home2d.domain.maps.build_map import random_world
from relpomdp.home2d.agent.transition_model import CanPickup, PickupEffect
from relpomdp.home2d.domain.env import Home2DEnvironment
from relpomdp.home2d.agent.transition_model import Pickup
import copy

def make_world():
    robot_id = 0
    init_robot_pose = (0, 0, 0)
    init_state, grid_map = random_world(6, 6, 3,
                                        ["Kitchen", "Office", "Office"],
                                        objects={"Office": {"Computer": (1, (1,1))},
                                                 "Kitchen": {"Salt": (1, (1,1)),
                                                             "Pepper": (1, (1,1))},
                                                 "Bathroom": {"Toilet": (1, (1,1))}},
                                        robot_id=robot_id, init_robot_pose=init_robot_pose,
                                        seed=10)
    env = Home2DEnvironment(robot_id,
                            grid_map,
                            init_state)
    return env



def test_mdp(env, nsteps=100, discount_factor=0.95):
    robot_id = env.robot_id
    init_robot_pose = env.robot_state["pose"]
    nk_agent = NKAgent(robot_id, init_robot_pose, grid_map=env.grid_map)
    fake_slam = FakeSLAM(Laser2DSensor(robot_id,
                                       fov=90, min_range=1,
                                       max_range=3, angle_increment=0.1))

    target_class = "Salt"
    target_id = list(env.ids_for(target_class))[0]
    init_belief = pomdp_py.Histogram({env.state.object_states[target_id]:1.0})
    nk_agent.add_target(target_id, target_class, init_belief)
    sensor = Laser2DSensor(robot_id,
                           fov=90, min_range=1,
                           max_range=2, angle_increment=0.1)
    nk_agent.add_sensor(sensor, {target_class: (10., 0.1)})
    nk_agent.update()

    agent = nk_agent.instantiate()
    env.set_reward_model(agent.reward_model)
    pickup_condeff = (CanPickup(env.robot_id, target_id), PickupEffect())
    env.transition_model.cond_effects.append(pickup_condeff)

    planner = pomdp_py.POUCT(max_depth=20,
                             discount_factor=discount_factor,
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
        if isinstance(action, Pickup):
            print("Done.")
            break
    viz.on_cleanup()
    return rewards

if __name__ == "__main__":
    env = make_world()
    test_mdp(env)
