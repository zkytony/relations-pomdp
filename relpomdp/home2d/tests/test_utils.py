from relpomdp.home2d.agent.transition_model import CanDeclareFound, DeclareFoundEffect, DeclareFound
from relpomdp.home2d.agent.reward_model import DeclareFoundRewardModel
from relpomdp.home2d.agent.policy_model import RandomPolicyModel, PreferredPolicyModel
from relpomdp.home2d.domain.env import Home2DEnvironment
from relpomdp.home2d.domain.maps.build_map import random_world
from relpomdp.oopomdp.framework import Objstate, OOState
import pomdp_py

def add_target(nk_agent, target_id, init_belief, env):
    """Adds a pick up action and effect for the given target;
    The agent will have a given initial belief about the target's
    location."""
    cond = CanDeclareFound(nk_agent.robot_id, target_id)
    # This effect will ONLY change the state of
    # the target specified by `target_id`
    effect = DeclareFoundEffect()
    action = DeclareFound()
    nk_agent.add_actions({action}, cond, effect)
    env.transition_model.cond_effects.append((cond, effect))

    reward_model = DeclareFoundRewardModel(nk_agent.robot_id, target_id)
    nk_agent.add_reward_model(reward_model)
    env.add_reward_model(reward_model)

    nk_agent.set_belief(target_id, init_belief)

def random_policy_model(nk_agent,
                        memory={},
                        actions=None):
    """Creates a RandomPolicyMOdel
    memory is a mapping from robot pose to actions,
    which indicates the set of valid actions at a pose.
    This memory can be expanded as the robot executes by
    remembering poses where not all actions can be taken
    for some reason"""
    if actions is None:
        actions = nk_agent.all_actions()
    policy_model = RandomPolicyModel(nk_agent.robot_id,
                                     actions,
                                     legal_motions=nk_agent.legal_motions,
                                     memory=memory)
    return policy_model

def preferred_policy_model(nk_agent,
                           action_prior_class,
                           ap_args=[],
                           ap_kwargs={},
                           memory={},
                           actions=None):
    if actions is None:
        actions = nk_agent.all_actions()
    policy_model = PreferredPolicyModel(nk_agent.robot_id,
                                        actions,
                                        legal_motions=nk_agent.legal_motions,
                                        memory=memory)
    policy_model.add_action_prior(action_prior_class, *ap_args, **ap_kwargs)
    return policy_model


def update_map(fake_slam, nk_agent, prev_robot_pose, robot_pose, env):
    """Updates the partial map maintained by the nk_agent,
    using the `fake_slam` map builder, based on previous and
    current robot poses and the full map provided by the environment.
    Also updates the nk_agent accordingly to maintain its integrity"""
    fake_slam.update(nk_agent.grid_map, prev_robot_pose, robot_pose, env)
    legal_motions = nk_agent.grid_map.compute_legal_motions(nk_agent.motion_actions)
    nk_agent.legal_motions = legal_motions
    nk_agent.move_condition.legal_motions = legal_motions
    nk_agent.check_integrity()

    # # After agent's map is updated, update the caches maintained by the agent;
    # # The pointers to these caches are used by the observe effect
    for sensor_name in nk_agent.sensor_caches:
        cache = nk_agent.sensor_caches[sensor_name]
        if cache.serving(nk_agent.grid_map):
            print("Updating caches for %s" % sensor_name)
            cache.update(nk_agent.grid_map)


def make_world(seed=100, worldsize=(6,6), init_robot_pose=(0,0,0), nrooms=3):
    """Creates a world for testing"""
    robot_id = 0
    w, l = worldsize
    init_state, grid_map = random_world(w, l, nrooms,
                                        ["Kitchen", "Office", "Office"],
                                        objects={"Office": {"Computer": (1, (1,1))},
                                                 "Kitchen": {"Salt": (1, (1,1)),
                                                             "Pepper": (1, (1,1))},
                                                 "Bathroom": {"Toilet": (1, (1,1))}},
                                        robot_id=robot_id, init_robot_pose=init_robot_pose,
                                        seed=seed)
    env = Home2DEnvironment(robot_id,
                            grid_map,
                            init_state)
    return env


def belief_fit_map(target_belief, updated_partial_map, **kwargs):
    """Given a target belief over a partial map, rescale it and add belief at
    additional locations in the updated partial map.

    The updated partial map usually should be an expansion of the partial map
    the `target_belief` is over, but could also be smaller, when some frontier
    is dropped due to blocked by wall.

    If `get_dict` is True in `kwargs`, return a dictionary from state to action.
    Otherwise, return pomdp_py.Histogram.
    """
    # Methodology:
    ### Belief at state B(s) = Val(s) / Norm, where Val is the unnormalized belief,
    ### and Norm is some normalizer. Here, we will regard the number of grid cells
    ### in a map as the normalizer, and compute the unnormalized belief accordingly.
    ### Basically we want to rescale the normalized belief to fit onto the updated map.
    updated_map_locations = updated_partial_map.frontier() | updated_partial_map.free_locations

    ## Belief update.
    cur_norm = len(target_belief)
    new_norm = len(updated_map_locations)

    new_norm_target_hist = {state:target_belief[state]*(cur_norm/new_norm) for state in target_belief}
    updated_total_prob = 1. - sum(new_norm_target_hist.values()) # The total unnormalized probability in the expanded region
    target_class = target_belief.random().objclass

    target_hist = {}
    for x, y in updated_map_locations:
        target_state = Objstate(target_class, pose=(x,y))
        if target_state in new_norm_target_hist:
            target_hist[target_state] = new_norm_target_hist[target_state]
        else:
            if new_norm < cur_norm:
                # Not going to track belief for this state. This state
                # should lie outside of the map boundary. You can pass
                # in a environment grid map for sanity check
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

    if kwargs.get("get_dict", False):
        return target_hist
    else:
        return pomdp_py.Histogram(target_hist)
