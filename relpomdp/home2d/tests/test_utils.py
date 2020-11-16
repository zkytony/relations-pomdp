from relpomdp.home2d.agent.transition_model import CanDeclareFound, DeclareFoundEffect, DeclareFound
from relpomdp.home2d.agent.reward_model import DeclareFoundRewardModel
from relpomdp.home2d.agent.policy_model import RandomPolicyModel, PreferredPolicyModel
from relpomdp.home2d.domain.env import Home2DEnvironment
from relpomdp.home2d.domain.maps.build_map import random_world
from relpomdp.oopomdp.framework import Objstate, OOState
from relpomdp.home2d.learning.generate_worlds import generate_world
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
    config = {
        "robot_id": 0,
        "init_robot_pose": (0, 0, 0),
        "width": worldsize[0],
        "length": worldsize[1],
        "nrooms": nrooms,
        "ndoors": 2,
        "room_types": ["Kitchen", "Office", "Office"],
        "objects": {"Office": {"Computer": (1, (1,1))},
                    "Kitchen": {"Salt": (1, (1,1)),
                                "Pepper": (1, (1,1))},
                    "Bathroom": {"Toilet": (1, (1,1))}},
        "shuffle_rooms": False,
        "min_room_size": 2,
        "max_room_size": 3,
    }
    return generate_world(config, seed=seed)


def belief_fit_map(target_belief, updated_partial_map, prev_partial_map, **kwargs):
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
    updated_frontier = updated_partial_map.frontier()
    prev_frontier = prev_partial_map.frontier()
    updated_map_locations = updated_frontier | updated_partial_map.free_locations

    belief_size = len(target_belief)
    next_belief_size = len(updated_map_locations)

    # Make sure target belief is normalized
    total_prob = sum(target_belief[s] for s in target_belief)
    assert abs(total_prob - 1.0) <= 1e-9

    # Update/Rescale existing beliefs
    target_hist = {}
    new_locations = set()
    prob_in_hist = 0.0
    for target_state in target_belief:
        if target_state["pose"] in prev_frontier:
            # This was previously a frontier location. If it still is, then rescale belief.
            # If it isn't on the frontier any more but a free location, then treat this location as a new location
            # to the map.
            if target_state["pose"] in updated_frontier:
                rescale = True
            else:
                rescale = False
                if target_state["pose"] in updated_partial_map.free_locations:
                    new_locations.add(target_state["pose"])
        else:
            if target_state["pose"] in updated_map_locations:
                # This was previously not a frontier - so it should still not be one.
                # We will rescale the belief at this location
                assert target_state["pose"] not in updated_frontier
                rescale = True

        if rescale:
            target_hist[target_state] = target_belief[target_state] * (belief_size / next_belief_size)
            prob_in_hist += target_hist[target_state]

    # Assign uniform complement belief over new locations.
    # New locations include: 1) brand new free grid cells,
    # 2) free grid cells converted from frontier grid cells (already in new_locations set)
    # 3) new frontier grid cells
    target_class = target_belief.random().objclass
    new_locations |= updated_map_locations - {s["pose"] for s in target_hist}
    for x, y in new_locations:
        target_state = Objstate(target_class, pose=(x,y))
        target_hist[target_state] = (1.0 - prob_in_hist) / len(new_locations)
        prob_in_hist += target_hist[target_state]

    # Renormalize
    prob_sum = sum(target_hist[state] for state in target_hist)
    for target_state in target_hist:
        target_hist[target_state] /= prob_in_hist

    if kwargs.get("get_dict", False):
        return target_hist
    else:
        return pomdp_py.Histogram(target_hist)
