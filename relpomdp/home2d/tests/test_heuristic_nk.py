# Random no knowledge agent
import pomdp_py
from sciex import Event
from relpomdp.home2d.agent import NKAgent, FakeSLAM, Laser2DSensor, NKAgentViz
from relpomdp.home2d.domain.condition_effect import MoveEffect
from relpomdp.home2d.tests.test_pomdp_nk import build_pomdp_nk_agent
from relpomdp.home2d.agent.transition_model import DeclareFound
from relpomdp.home2d.constants import FILE_PATHS
from relpomdp.home2d.utils import euclidean_dist
from relpomdp.home2d.tests.test_utils import add_target, random_policy_model, make_world, update_map
from relpomdp.home2d.learning.generate_worlds import make_graph
from collections import deque
import copy
import random
import time

def build_heuristic_nk_agent(env, target_class,
                             target_sensor_config={},
                             slam_sensor_config={}):
    """Build Random agent, with no knowledge"""
    nk_agent, fake_slam = build_pomdp_nk_agent(env, target_class,
                                               target_sensor_config=target_sensor_config,
                                               slam_sensor_config=slam_sensor_config)
    return nk_agent, fake_slam

def heuristic_action_selection(nk_agent, robot_pose, motion_queue, visit_counts={}):
    """Select a frontier location closest to the robot. Choose an
    action that brings the robot closer to that frontier location."""
    if len(motion_queue) > 0:
        # This chosen motion may not be legal motion (i.e. could bump into wall)
        # because when it is planned, the frontier could be actually blocked
        # by a wall but the robot didn't know that. If blocked, then the chosen
        # motion won't be a legal motion and we will generate new motions.
        chosen_motion = motion_queue.popleft()
        if chosen_motion in nk_agent.legal_motions[robot_pose[:2]]:
            return chosen_motion

    # Generate new motions.
    # If frontier is significantly large, find the shortest path to
    # a random frontier location and plan to get there.
    partial_map = nk_agent.grid_map
    frontier = partial_map.frontier()
    if len(frontier) >= 5:
        while len(motion_queue) == 0:
            frontier_point = random.sample(frontier, 1)[0]
            graph = make_graph(nk_agent.legal_motions,
                               free_locations=partial_map.free_locations | {frontier_point})
            try:
                path = graph.dijkstra(robot_pose[:2], frontier_point)
                for nid, eid in path:
                    if eid not in graph.edges_from(nid):
                        import pdb; pdb.set_trace()
                    motion = graph.edges[eid].data
                    motion_queue.append(motion)
            except KeyError:
                print("Failed to find path to frontier")
        chosen_motion = motion_queue.popleft()
    else:
        # We don't have a good motion to explore the map. Then,
        # among the neighbors, pick the one with lowest visit count
        chosen_motion = None,
        lowest = float('inf')
        # Legal motions at the robot's current location
        legal_motions = list(nk_agent.legal_motions[robot_pose[:2]])
        for motion in legal_motions:
            next_robot_pose = MoveEffect.move_by(robot_pose, motion)
            if next_robot_pose[:2] not in visit_counts:
                visit_counts[next_robot_pose[:2]] = 0
            count = visit_counts[next_robot_pose[:2]]
            if count < lowest:
                lowest = count
                chosen_motion = motion
    return chosen_motion

def step_heuristic_nk(env, nk_agent, fake_slam, target_id,
                      declare_next=False, visit_counts={},
                      motion_queue=deque([]),
                      true_pos_rate=1.0):
    """Runs a step in the MDP simulation"""
    policy_model = random_policy_model(nk_agent)
    agent = nk_agent.instantiate(policy_model)

    # Plan action
    prev_robot_pose = agent.belief.mpe().object_states[env.robot_id]["pose"]
    if prev_robot_pose[:2] not in visit_counts:
        visit_counts[prev_robot_pose[:2]] = 0
    visit_counts[prev_robot_pose[:2]] += 1

    if declare_next:
        action = DeclareFound()
    else:
        action = heuristic_action_selection(nk_agent, prev_robot_pose, motion_queue,
                                            visit_counts=visit_counts)
    time.sleep(0.2)

    # environment transitions and obtains reward (note that we use agent's reward model for convenience)
    env_state = env.state.copy()
    _ = env.state_transition(action, execute=True)
    env_next_state = env.state.copy()
    reward = agent.reward_model.sample(env_state, action, env_next_state)

    observation = agent.observation_model.sample(env.state, action)

    # Updates
    ## update belief of robot
    new_robot_belief = pomdp_py.Histogram({env.robot_state.copy() : 1.0})
    nk_agent.set_belief(env.robot_id, new_robot_belief)
    robot_pose = new_robot_belief.mpe()["pose"]

    # update map (fake slam)
    update_map(fake_slam, nk_agent, prev_robot_pose, robot_pose, env)

    # See if declare next
    if observation.object_observations[target_id]["pose"] is not None:
        if robot_pose[:2] == observation.object_observations[target_id]["pose"]:
            if random.uniform(0,1) < true_pos_rate:
                # Sensor functioning - believes in observation
                declare_next = True
    return action, copy.deepcopy(env.state), observation, reward, declare_next


def test_heuristic_nk(env, target_class,
                      discount_factor=0.95,
                      nsteps=100, save=False,
                      target_sensor_config={},
                      slam_sensor_config={},
                      visualize=True,
                      logger=None, **kwargs):
    robot_id = env.robot_id
    target_id = list(env.ids_for(target_class))[0]

    nk_agent, fake_slam = build_heuristic_nk_agent(env, target_class,
                                                   target_sensor_config=target_sensor_config,
                                                   slam_sensor_config=slam_sensor_config)
    motion_queue = deque([])
    visit_counts = {}

    # policy_model = heuristic_policy_model(nk_agent)
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
    declare_next = False
    for i in range(nsteps):
        # Visualize
        if visualize:
            viz.on_loop()
            img_agent, img_world, img = viz.on_render()
            game_states.append(img)

        # Take a step
        if len(target_sensor_config) > 0:
            true_pos_rate, false_pos_rate = target_sensor_config["noises"]
        else:
            target_sensor_name = list(nk_agent.sensors_for(target_class))[0]
            noises = nk_agent.sensors[target_sensor_name][1]
            true_pos_rate, false_pos_rate = noises[target_class]

        action, next_state, observation, reward, declare_next = \
            step_heuristic_nk(env, nk_agent, fake_slam, target_id,
                              motion_queue=motion_queue,
                              visit_counts=visit_counts,
                              true_pos_rate=true_pos_rate,
                              declare_next=declare_next)

        # Info and logging
        _disc_reward += _gamma*reward
        _gamma *= discount_factor
        _step_info = "Step {} : Action: {}    Reward: {}    DiscCumReward: {:.4f}    RobotPose: {}   TargetFound: {}"\
            .format(i+1, action, reward, _disc_reward,
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
        game_states.append(img_world)
        viz.on_cleanup()

        if save:
            print("Saving images...")
            dirp = "./demos/test_heuristic_nk"
            save_images_and_compress(game_states,
                                     dirp)
            subprocess.Popen(["nautilus", dirp])
    return _rewards, _states, _history

if __name__ == "__main__":
    # To test an ordinary run: set seed to be 100. init robot pose (0,0,0)
    # To test the 'unable to see wall next to robot' issue, set seed to 1459,
    #    set init robot pose (0,1,0). Try a few times because doorway may open up differently
    env = make_world(seed=1459, worldsize=(6,6))
    test_heuristic_nk(env, target_class="Salt", save=False, nsteps=100)
