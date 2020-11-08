from relpomdp.home2d.agent.transition_model import CanDeclareFound, DeclareFoundEffect, DeclareFound
from relpomdp.home2d.agent.reward_model import DeclareFoundRewardModel
from relpomdp.home2d.agent.policy_model import RandomPolicyModel, PreferredPolicyModel
from relpomdp.home2d.domain.env import Home2DEnvironment
from relpomdp.home2d.domain.maps.build_map import random_world

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
