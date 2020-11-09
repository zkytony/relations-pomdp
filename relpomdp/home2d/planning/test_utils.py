import pomdp_py
from relpomdp.home2d.tests.test_utils import\
    add_target, random_policy_model, make_world, update_map,\
    preferred_policy_model, belief_fit_map
from relpomdp.home2d.agent.reward_model import ReachRewardModel
from relpomdp.oopomdp.framework import Objstate

def remap(oldval, oldmin, oldmax, newmin, newmax):
    return (((oldval - oldmin) * (newmax - newmin)) / (oldmax - oldmin)) + newmin

def add_reach_target(nk_agent, target_id, init_belief):
    """
    Adds a subgoal to reach the target - By reach, it means the
    robot is directly on top of the target.
    """
    reward_model = ReachRewardModel(nk_agent.robot_id, target_id)
    nk_agent.add_reward_model(reward_model)
    nk_agent.set_belief(target_id, init_belief)
    return reward_model

def add_room_states(env, starting_room_id=10000):
    """Given environemnt, add in its state space a
    state for each room, which is located at one of the room's doorways"""
    # We will add a state per doorway per room
    room_id = starting_room_id
    for room_name in env.grid_map.rooms:
        room = env.grid_map.rooms[room_name]
        for doorway in room.doorways:
            room_state = Objstate(room.room_type,
                                  pose=doorway)
            env.add_object_state(room_id, room_state)
            room_id += 100
        room_id += 1000

def difficulty(df_difficulty, objclass):
    try:
        return float(df_difficulty.loc[df_difficulty["class"] == objclass]["difficulty"])
    except TypeError:
        return 1000

def correlation(df_corr, class1, class2):
    try:
        return float(df_corr.loc[(df_corr["class1"] == class1)\
                                 & (df_corr["class2"] == class2)]["corr_score"])
    except TypeError:
        return 0
