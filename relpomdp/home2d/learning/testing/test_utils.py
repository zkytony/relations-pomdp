import pomdp_py
from relpomdp.home2d.agent.tests.test_utils import add_pickup_target, random_policy_model, make_world, update_map,\
    preferred_policy_model
from relpomdp.home2d.agent.reward_model import ReachRewardModel

def add_reach_target(nk_agent, target_id, init_belief):
    """
    Adds a subgoal to reach the target - By reach, it means the
    robot is directly on top of the target.
    """
    reward_model = ReachRewardModel(nk_agent.robot_id, target_id)
    nk_agent.add_reward_model(reward_model)
    nk_agent.set_belief(target_id, init_belief)
    return reward_model
