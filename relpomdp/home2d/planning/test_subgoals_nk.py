# Test an agent that dynamically generates subgoal object
# to search for besides the target object, and uses correlation
# to update belief.

import pomdp_py
from relpomdp.home2d.agent import *
from relpomdp.home2d.domain import *
from relpomdp.home2d.utils import save_images_and_compress, discounted_cumulative_reward, euclidean_dist
from relpomdp.home2d.learning.generate_worlds import generate_world
from relpomdp.home2d.learning.correlation_observation_model\
    import compute_detections, CorrelationObservationModel
from relpomdp.oopomdp.framework import OOState, OOBelief
from relpomdp.home2d.constants import FILE_PATHS
from relpomdp.home2d.planning.test_utils import add_reach_target, difficulty, correlation,\
    add_target, preferred_policy_model, update_map, belief_fit_map
import copy
import time
import subprocess
import argparse
import yaml
import math
import pandas as pd


def select_subgoal(df_subgoal, target_class, excluded_classes=set()):
    """Returns a class that is the best subgoal to search for the target class,
    based on the scores given in `df_subgoal`."""
    subdf = df_subgoal.loc[df_subgoal["c1"] == target_class]
    best_score = float("-inf")
    best_subgoal = None
    for subgoal_class in subdf["c2"]:
        if subgoal_class in excluded_classes:
            continue
        score = float(subdf.loc[subdf["c2"] == subgoal_class]["score"])
        if score > best_score:
            best_score = score
            best_subgoal = subgoal_class
    return best_subgoal

def subgoal_sequence(target_class,
                     df_subgoal, df_difficulty,
                     difficulty_threshold="Kitchen"):
    """Utility function that returns a sequence of subgoals needed
    to find the given target class. Not actually used by any other
    function in this file."""
    if type(difficulty_threshold) == str:
        difficulty_threshold = difficulty(df_difficulty, difficulty_threshold)
    task_difficulty = difficulty(df_difficulty, target_class)
    subgoals = [target_class]
    excluded_classes = set()
    while task_difficulty > difficulty_threshold:
        # Select subgoal
        subgoal_class = select_subgoal(df_subgoal, subgoals[-1],
                                       excluded_classes=excluded_classes)
        subgoals.append(subgoal_class)
        # Recompute difficulty
        task_difficulty = difficulty(df_difficulty, subgoal_class)
        excluded_classes.add(subgoal_class)
    return subgoals


def uniform_belief(objclass, nk_agent):
    obj_hist = {}
    total_prob = 0.0
    partial_map = nk_agent.grid_map
    robot_state = nk_agent.object_beliefs[nk_agent.robot_id].mpe()
    for x, y in partial_map.free_locations | partial_map.frontier():
        obj_state = Objstate(objclass, pose=(x,y))
        if (x,y) == robot_state["pose"][:2]:
            obj_hist[obj_state] = 1e-12
        else:
            obj_hist[obj_state] = 1.
        total_prob += obj_hist[obj_state]
    for state in obj_hist:
        obj_hist[state] /= total_prob
    init_belief = pomdp_py.Histogram(obj_hist)
    return init_belief


def search(target_class, target_id, nk_agent, fake_slam, env, viz,
           df_difficulty, df_corr, df_subgoal,
           difficulty_threshold="Kitchen",
           use_correlation_belief_update=True,
           logger=None,
           **kwargs):
    """
    Searches for an instance of the given target class.
    If the target class is difficult to be found, then
    generates a subgoal class, and call this function recursively.
    Otherwise, proceed to take actions to find the target.

    The given nk_agent should have all sensors equipped.
    The given agent should already been told the final goal class,
    and the `env` should also already contain reward model to measure
    the global goal.
    """
    # Print info
    _search_info = "\n"
    _search_info += "Target class: %s\n" % target_class
    _search_info += "Searcher parameters:\n"
    _search_info += "    max_depth: %d\n" % kwargs.get("max_depth", -1)
    _search_info += "    num_sims: %d\n" % kwargs.get("num_sims", -1)
    _search_info += "    discount_factor: %f\n" % kwargs.get("discount_factor", 0.95)
    _search_info += "    exploration const: %f\n" % kwargs.get("exploration_constant", 200)
    nsteps = kwargs.get("nsteps", 100)
    _search_info += "    num steps allowed: %d" % nsteps
    if logger is None:
        print(_search_info)
    else:
        logger(_search_info)

    if type(difficulty_threshold) == str:
        difficulty_threshold = difficulty(df_difficulty, difficulty_threshold)

    # Build subgaols
    task_difficulty = difficulty(df_difficulty, target_class)
    subgoals = [(target_class, target_id)]
    excluded_classes = {target_class}
    while task_difficulty > difficulty_threshold:
        # Select subgoal
        subgoal_class = select_subgoal(df_subgoal, subgoals[-1][0],
                                       excluded_classes=excluded_classes)
        if subgoal_class is None:
            _info = "No more subgoal generatable. Proceed search."
            if logger is None:
                print(_info)
            else:
                logger(_info)
            break

        # Recompute difficulty
        new_task_difficulty = difficulty(df_difficulty, subgoal_class)
        excluded_classes.add(subgoal_class)

        # Check and accept subgoal
        if new_task_difficulty < task_difficulty:
            subgoal_id = list(env.ids_for(subgoal_class))[0]
            subgoals.append((subgoal_class, subgoal_id))
            _info = "Selected subgoal class: %s" % subgoal_class
            if logger is None:
                print(_info)
            else:
                logger(_info)

            # Add the subgoal to nk_agent; So the agent is aware of
            # all of them in its reward model when planning.
            init_belief = uniform_belief(subgoal_class, nk_agent)
            add_reach_target(nk_agent, subgoal_id, init_belief)

        # Update task difficulty
        task_difficulty = new_task_difficulty

    # Start with the last subgoal
    all_reaching_goals = subgoals[1:]
    _rewards = []
    _states = [copy.deepcopy(env.state)]
    _history = []
    _disc_reward = [0.0]  # trick
    _gamma = [1.0]
    nsteps_remaining = nsteps
    while len(subgoals) > 0:
        subgoal_class, subgoal_id = subgoals[-1]
        reaching = subgoal_class != target_class
        kwargs["nsteps"] = nsteps_remaining
        subgoals_done, steps_taken =\
            _run_search(nk_agent, subgoal_class, subgoal_id,
                        df_corr, fake_slam, env, viz,
                        reaching=reaching, all_reaching_goals=all_reaching_goals,
                        use_correlation_belief_update=use_correlation_belief_update,
                        logger=logger,
                        # for logging (will modify)
                        _rewards=_rewards, _states=_states, _history=_history,
                        _gamma=_gamma, _disc_reward=_disc_reward,
                        **kwargs)
        nsteps_remaining -= steps_taken
        if subgoals_done is not None:
            # subgoals_done should be a set of object ids
            subgoals = [tup for tup in subgoals
                        if tup[1] not in subgoals_done]
            all_reaching_goals = subgoals[1:]
        if nsteps_remaining == 0:
            break

    if viz is not None:
        viz.on_cleanup()
    return _rewards, _states, _history

def _run_search(nk_agent, target_class, target_id,
                df_corr, fake_slam, env, viz,
                reaching=False, all_reaching_goals=[],
                use_correlation_belief_update=True,
                visualize=False, logger=None,
                # for logging (will modify)
                _rewards=None, _states=None, _history=None,
                _gamma=None, _disc_reward=None,
                **kwargs):
    """Runs the agent to search for target. By 'search', I mean
    that the target can be found by 'reaching' (if it's a subgoal),
    or by 'pickup' (if it's the goal object). TODO: FIX THIS COMMENT (no pickup any more)

    reaching is True if the target is found by reaching to its location (instead
    of picking up)

    all_reaching_goals is a list of subgoals that can be achieved by reaching.
    Once reached, this subgoal will be considered completed and its reward will
    be removed from the agent.

    Returns:
        subgoals_done (set of object ids that correspond to subgoals that are done by reaching)
        rewards (list of rewards collected)
    """
    # Build an agent
    sensor_names = set()
    objects_tracking = set({nk_agent.robot_id})
    for objclass, objid in [(target_class, target_id)] + all_reaching_goals:
        sensor_names.update(nk_agent.sensors_for(objclass))
        objects_tracking.add(objid)

    actions = set(nk_agent.motion_actions)
    if not reaching:
        # We also have picking
        actions.add(DeclareFound())
    policy_model = preferred_policy_model(nk_agent,
                                          GreedyActionPrior,
                                          ap_args=[target_id],
                                          actions=actions)
    planning_agent = nk_agent.instantiate(policy_model,
                                          sensors_in_use=sensor_names,
                                          objects_tracking=objects_tracking)
    # Build the sensor observation model
    # env grid map observation model has its own cache (that needs no update).
    caches = {}
    for name in nk_agent.sensors:
        caches[name] = SensorCache(name)
        caches[name].serving(env.grid_map.name)
    observation_model = nk_agent.build_observation_model(grid_map=env.grid_map,
                                                         caches=caches)

    # build the correlation observation model
    room_types = set(env.grid_map.rooms[name].room_type
                     for name in env.grid_map.rooms)
    corr_obs_model = CorrelationObservationModel(nk_agent.robot_id,
                                                 room_types,
                                                 df_corr)

    _depth = kwargs.get("max_depth", 20)
    _discount_factor = kwargs.get("discount_factor", 0.95)
    _num_sims = kwargs.get("num_sims", 600)
    _exploration_constant = kwargs.get("exploration_constant", 200)
    planner = pomdp_py.POUCT(max_depth=_depth,
                             discount_factor=_discount_factor,
                             num_sims=_num_sims,
                             exploration_const=_exploration_constant,  # todo: is 100 still good?
                             rollout_policy=planning_agent.policy_model)

    _nsteps = kwargs.get("nsteps", 100)
    for step in range(_nsteps):

        if viz is not None:
            viz.on_loop()
            img_agent, img_world, img = viz.on_render(OOBelief(nk_agent.object_beliefs))

        # Plan action
        start_time = time.time()
        action = planner.plan(planning_agent)

        if logger is None:
            print("__ (Step %d/%d) Searching for %s, %d ____" % (step+1, _nsteps, target_class, target_id))
            print("#### POUCT (took %.4fs) ####" % (time.time() - start_time))
            planner.print_action_values()

        # Environment transition
        env_state = env.state.copy()
        prev_robot_pose = planning_agent.belief.mpe().object_states[nk_agent.robot_id]["pose"]
        _ = env.state_transition(action, execute=True)
        env_next_state = env.state.copy()
        reward = env.reward_model.sample(env_state, action, env_next_state)

        # Get observation using all sensors
        observation = observation_model.sample(env.state, action)
        detected_classes, detected_ids, detected_poses = compute_detections(observation, return_poses=True)

        # update belief of robot
        new_robot_belief = pomdp_py.Histogram({env.robot_state.copy() : 1.0})
        robot_state = new_robot_belief.mpe()

        # update map (fake slam)
        update_map(fake_slam, nk_agent, prev_robot_pose, robot_state["pose"], env)

        partial_map = nk_agent.grid_map
        map_locations = partial_map.frontier() | partial_map.free_locations

        # update belief about all objects in nk_agent independently
        for objid in nk_agent.object_beliefs:
            if objid == nk_agent.robot_id:
                continue
            # First, expand the belief space to cover the expanded map
            obj_belief = nk_agent.object_beliefs[objid]
            obj_hist = belief_fit_map(obj_belief, nk_agent.grid_map,
                                      get_dict=True)

            # Then, perform belief update
            next_obj_hist = {}
            total_prob = 0.0
            for obj_state in obj_hist:
                oostate = OOState({nk_agent.robot_id: robot_state,
                                   objid: obj_state})
                obs_prob = observation_model.probability(observation, oostate, action)

                if use_correlation_belief_update:
                    corr_prob = corr_obs_model.probability(observation, oostate, action,
                                                           objid=objid, grid_map=partial_map)

                    next_obj_hist[obj_state] = corr_prob * obs_prob * obj_hist[obj_state]  # static objects
                else:
                    next_obj_hist[obj_state] = obs_prob * obj_hist[obj_state]  # static objects

                total_prob += next_obj_hist[obj_state]
            for obj_state in next_obj_hist:
                next_obj_hist[obj_state] /= total_prob

            nk_agent.set_belief(objid, pomdp_py.Histogram(next_obj_hist))
        nk_agent.set_belief(nk_agent.robot_id, new_robot_belief)

        # Make new agents because map is updated; Just calling the same functions
        nk_agent.check_integrity()
        policy_model = preferred_policy_model(nk_agent,
                                              GreedyActionPrior,
                                              ap_args=[target_id],
                                              actions=actions)
        planning_agent = nk_agent.instantiate(policy_model,
                                              sensors_in_use=sensor_names,
                                              objects_tracking=objects_tracking)
        planner.set_rollout_policy(planning_agent.policy_model)

        # Info and logging
        next_state = copy.deepcopy(env.state)
        _disc_reward[0] += _gamma[0]*reward
        _gamma[0] *= _discount_factor
        _step_info = "Step {}/{} : SearchingFor: {}    Action: {}    NumSims:  {}    Reward: {}    DiscCumReward: {:.4f}    "\
            "Detections: {}    RobotPose: {}   TargetFound: {}"\
            .format(step+1, _nsteps, target_class, action, planner.last_num_sims,
                    reward, _disc_reward[0], detected_classes,
                    next_state.object_states[env.robot_id]["pose"],
                    next_state.object_states[target_id].get("is_found", False))
        if logger is None:
            print(_step_info)
        else:
            logger(_step_info)

        _rewards.append(reward)
        _states.append(next_state)
        _history.append((action, observation, copy.deepcopy(nk_agent.object_beliefs)))

        # Check termination
        subgoal_ids = set(subgoal_id for _, subgoal_id in all_reaching_goals)
        subgoal_classes = set(subgoal_class for subgoal_class, _ in all_reaching_goals)
        subgoals_done = set()
        ## Subgoal is finished if we have detected an object of that class and
        ## the pose of the detection is close to the robot. (No check by id - so
        ## this won't work if the robot sets a second subgoal of finding an
        ## object of the same category).
        for subgoal_class, subgoal_id in all_reaching_goals:
            if subgoal_class in detected_classes:
                if euclidean_dist(robot_state["pose"][:2], detected_poses[subgoal_class]) <= 1:
                    # if robot_state["pose"][:2] == env.state.object_states[subgoal_id]["pose"]:
                    subgoals_done.add(subgoal_id)
                    _info = "Subgoal %s, %d is done!" % (subgoal_class, subgoal_id)
                    if logger is None:
                        print(_info)
                    else:
                        logger(_info)

        # Remove reward for done subgoals
        for objid in subgoals_done:
            nk_agent.remove_reward_model(objid)
        # If our purpose is to reach a certain class, then we can return if it is done
        if reaching:
           if target_class in detected_classes:
               return subgoals_done, step+1
        else:
            # We are picking
            if isinstance(action, DeclareFound):
                if logger is None:
                    print("Done!")
                else:
                    logger("Done!")
                return [target_id], step+1
    return None, _nsteps


def test_subgoals_agent(env, target_class, config,
                        df_corr, df_dffc, df_subgoal,
                        difficulty_threshold="Kitchen",
                        nsteps=100, discount_factor=0.95, max_depth=15,
                        num_sims=300, exploration_constant=100,
                        use_correlation_belief_update=True,
                        target_sensor_config={},
                        slam_sensor_config={},
                        full_map=False,
                        visualize=True,
                        logger=None):
    """The function to call.

    Args:
        env: The environment
        target_class (str): object class to search for (and pickup)  # TODO: FIX THIS COMMENT (no pickup anymore)
        config (dict): Configurations, read from a config file
        df_corr
        df_dffc
        df_subgoal: These three are pandas dataframes that store learned scores
        difficulty_threshold (str): The class that if difficulty is above the difficulty
            of searching for this class, then will generate a subgoal.
    """
    robot_id = env.robot_id
    target_id = list(env.ids_for(target_class))[0]

    # Build an NKAgent, equipped with all sensors
    if full_map:
        nk_agent = NKAgent(env.robot_id, env.robot_state["pose"], grid_map=env.grid_map)
    else:
        nk_agent = NKAgent(env.robot_id, env.robot_state["pose"])

    for sensor_name in config["sensors"]:
        cfg = config["sensors"][sensor_name]
        noises = cfg["noises"]
        if target_class in noises and len(target_sensor_config) > 0:
            cfg = target_sensor_config
        sensor = Laser2DSensor(env.robot_id,
                               name=sensor_name,
                               fov=float(cfg["fov"]),
                               min_range=float(cfg["min_range"]),
                               max_range=float(cfg["max_range"]),
                               angle_increment=float(cfg["angle_increment"]))
        nk_agent.add_sensor(sensor, noises)

    # Create visualization
    with open(FILE_PATHS["colors"]) as f:
        colors = yaml.load(f)
        for objclass in colors:
            colors[objclass] = pomdp_py.util.hex_to_rgb(colors[objclass])

    viz = None
    if visualize:
        viz = NKAgentViz(nk_agent,
                         env,
                         colors,
                         res=30,
                         controllable=True,
                         img_path=FILE_PATHS["object_imgs"])
        viz.on_init()

    # SLAM sensor has same range as room sensor
    if slam_sensor_config is None:
        room_sensor = nk_agent.sensors[list(nk_agent.sensors_for("Kitchen"))[0]][0]
        slam_sensor_config = {
            "fov": room_sensor.fov,
            "min_range": room_sensor.min_range,
            "max_range": room_sensor.max_range,
            "angle_increment": room_sensor.angle_increment,
        }

    fake_slam = FakeSLAM(Laser2DSensor(robot_id,
                                       fov=slam_sensor_config.get("fov", 360),
                                       min_range=slam_sensor_config.get("min_range", 1),
                                       max_range=slam_sensor_config.get("max_range", 4),
                                       angle_increment=slam_sensor_config.get("angle_increment", 0.1)))

    # We simulate that the robot first stays in place to map what's in front,
    # and then rotates 90 degrees to map what's on top. This gives the robot
    # a small initial map to work on instead of just its grid cell.
    init_robot_pose = env.robot_state["pose"]
    update_map(fake_slam, nk_agent, init_robot_pose, init_robot_pose, env)
    rotated_robot_pose = (init_robot_pose[0], init_robot_pose[1], init_robot_pose[2] + math.pi/2)
    update_map(fake_slam, nk_agent, init_robot_pose, rotated_robot_pose, env)

    # Tell the agent that your task is to pick up the target object class
    init_belief = uniform_belief(target_class, nk_agent)
    add_target(nk_agent, target_id, init_belief, env)

    rewards = search(target_class, target_id, nk_agent, fake_slam, env, viz,
                     df_dffc, df_corr, df_subgoal,
                     difficulty_threshold=difficulty_threshold,
                     discount_factor=discount_factor,
                     exploration_constant=exploration_constant,
                     max_depth=max_depth,
                     num_sims=num_sims,
                     nsteps=nsteps,
                     use_correlation_belief_update=use_correlation_belief_update,
                     visualize=visualize,
                     logger=logger)
    return rewards


def main():
    parser = argparse.ArgumentParser(description="Run the object search with subgoals program.")
    parser.add_argument("config_file",
                        type=str, help="Path to .yaml configuration file (world distribution)")
    parser.add_argument("diffc_score_file",
                        type=str, help="Path to .csv for difficulty")
    parser.add_argument("corr_score_file",
                        type=str, help="Path to .csv for correlation")
    parser.add_argument("subgoal_score_file",
                        type=str, help="Path to .csv for subgoal selection")
    parser.add_argument("-T", "--target-class", default="Salt",
                        type=str, help="Target class to search for")
    parser.add_argument("--seed",
                        type=int, help="Seed for world generation")
    args = parser.parse_args()

    with open(args.config_file) as f:
        config = yaml.load(f)

    print("Generating environment that surely contains %s" % args.target_class)
    seed = args.seed
    env = generate_world(config, seed=seed, required_classes={args.target_class})

    test_subgoals_agent(env, args.target_class, config,
                        df_corr=pd.read_csv(args.corr_score_file),
                        df_dffc=pd.read_csv(args.diffc_score_file),
                        df_subgoal=pd.read_csv(args.subgoal_score_file),
                        use_correlation_belief_update=True,
                        full_map=False)

if __name__ == "__main__":
    main()


#python test_subgoals_nk.py ../configs/10x10_10-20-2020.yaml ../data/difficulty-try1-10-20-2020-20201026162744897.csv ../data/correlation-try1-10-20-2020.csv scores.csv -T Salt
