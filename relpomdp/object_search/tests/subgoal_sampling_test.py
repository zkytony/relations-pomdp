import pomdp_py
import copy
from relpomdp.oopomdp.abstraction import AbstractAttribute
from relpomdp.object_search.world_specs.build_world import *
import relpomdp.oopomdp.framework as oopomdp
from relpomdp.planning.subgoal import Subgoal
from relpomdp.object_search import *
import time


if __name__ == "__main__":
    # Test
    ids, grid_map, init_state, colors = salt_pepper_1()

    # Obtain prior
    target_hist = {}
    total_prob = 0
    for x in range(grid_map.width):
        for y in range(grid_map.length):
            state = ItemState("Salt", (x,y))
            target_hist[state] = 1.0
            total_prob += target_hist[state]
    # Normalize
    for state in target_hist:
        target_hist[state] /= total_prob
    
    robot_id = ids["Robot"]
    target_id = ids["Target"][0]    
    env = ObjectSearchEnvironment(ids,
                                  grid_map,
                                  init_state)
    sensor = Laser2DSensor(robot_id, env.grid_map, fov=90, min_range=1, max_range=2,
                           angle_increment=0.5)
    # Build a regular agent
    init_belief = oopomdp.OOBelief({target_id: pomdp_py.Histogram(target_hist),
                                     robot_id: pomdp_py.Histogram({env.robot_state:1.0})},
                                   oo_state_class=JointState)
    agent = ObjectSearchAgent(env.grid_map, sensor, env.ids,
                              init_belief)

    # Build the agent used for subgoal planning
    sg = ReachRoomSubgoal(ids, "Kitchen", grid_map)
    subgoals = {sg.name: sg}
    robot_id = ids["Robot"]
    robot_state = agent.belief.mpe().object_states[robot_id]    
    robot_state_with_subgoals =\
        RobotStateWithSubgoals.from_state_without_subgoals(
            robot_state, subgoals=tuple((sg_name, Subgoal.IP)
                                        for sg_name in subgoals))
    belief = oopomdp.OOBelief({robot_id:pomdp_py.Histogram({robot_state_with_subgoals.copy():1.0}),
                               target_id:agent.belief.object_beliefs[target_id]},
                              oo_state_class=JointState) 
    transition_model = oopomdp.OOTransitionModel(
        set(agent.transition_model.cond_effects)\
        | {(AchievingSubgoals(ids, subgoals),
            UpdateSubgoalStatus(ids))})
    reward_model = SubgoalRewardModel(ids)
    tmp_agent = pomdp_py.Agent(belief,
                               agent.policy_model,
                               transition_model,
                               agent.observation_model,
                               reward_model)

    start = time.time()
    for i in range(100):
        next_state, observation, reward, _ = pomdp_py.sample_generative_model(
            tmp_agent, tmp_agent.belief.mpe(), MoveW)
    total_time = time.time() - start
    print("Sampling generative model 100 times used %.9fs" % total_time)
    print(next_state)
