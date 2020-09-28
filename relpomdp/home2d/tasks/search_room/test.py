from relpomdp.home2d.tasks.search_room.search_room_task import *
from relpomdp.home2d.domain.visual import Home2DViz
from relpomdp.home2d.domain.maps import all_maps
import relpomdp.oopomdp.framework as oopomdp
import time


def setup():
    grid_map = all_maps["map_small_1"]()

    # Building object state
    robot_id = 1
    robot_state = objstate("Robot",
                           pose=(0,0,0),
                           camera_direction="+x",
                           room_type=grid_map.room_of((0,0)).room_type)

    # Make task
    task = SearchRoomTask(robot_id, "Office", grid_map=grid_map, knows_room_type=True)
    
    # Make environment
    init_state = {robot_id: robot_state}
    env = Home2DEnvironment(robot_id,
                            grid_map, init_state,
                            reward_model=task.reward_model)
    env.transition_model.cond_effects.append((
        (CanMove(robot_id, env.legal_motions), MoveEffectRoom(robot_id, grid_map))
    ))

    # Initialize belief: uniform over room types
    room_types = set(env.grid_map.rooms[r].room_type
                     for r in grid_map.rooms)
    init_belief = pomdp_py.OOBelief({robot_id: pomdp_py.Histogram({
        objstate("Robot",
                 pose=robot_state["pose"],
                 camera_direction=robot_state["camera_direction"],
                 room_type=room_type) : 1.0 / len(room_types)
        for room_type in room_types
    })})
    agent = task.to_agent(init_belief)

    # Create planner and make a plan
    planner = pomdp_py.POUCT(max_depth=30,
                             discount_factor=0.95,
                             num_sims=500,
                             exploration_const=50,
                             rollout_policy=agent.policy_model)

    viz = Home2DViz(env,
                    {},
                    res=30)
    
    return env, agent, task, planner, viz


def solve(env, agent, task, planner, viz):
    viz.on_init()
    viz.on_render()

    for step in range(100):
        print("---- Step %d ----" % step)
        action = planner.plan(agent)
        reward = env.state_transition(action, execute=True)
        observation = agent.observation_model.sample(env.state, action)


        # Belief update
        robot_state = env.state.object_states[task.robot_id]
        room_types = set(env.grid_map.rooms[r].room_type
                         for r in env.grid_map.rooms)
        # Create a next state space on top of new robot pose, and all room types.
        next_state_space = {
            oopomdp.OOState({task.robot_id: objstate("Robot",
                                                     pose=robot_state["pose"],
                                                     camera_direction=robot_state["camera_direction"],
                                                     room_type=room_type)})
            for room_type in room_types
        }
        # Need to get the current belief to be regarding OOState (for the calls to T/R/O models to work)
        cur_belief = pomdp_py.Histogram({
            oopomdp.OOState({task.robot_id: state}) : agent.belief.object_beliefs[task.robot_id][state]
            for state in agent.belief.object_beliefs[task.robot_id]
        })
        new_belief = pomdp_py.update_histogram_belief(cur_belief,
                                                      action, observation,
                                                      agent.observation_model,
                                                      agent.transition_model,
                                                      next_state_space=next_state_space)
        # Take just the target state from this
        new_belief = pomdp_py.Histogram({state.object_states[task.robot_id]:
                                         new_belief[state]
                                         for state in new_belief})
        agent.belief.set_object_belief(task.robot_id, new_belief)
        planner.update(agent, action, observation)
        
        print("     action: %s" % str(action.name))        
        print("     reward: %s" % str(reward))
        print("observation: %s" % str(observation))
        print("robot state: %s" % str(robot_state))

        time.sleep(1)
        
        viz.on_loop()
        viz.on_render()

        if action == Stop() and env.robot_state["room_type"] == task.room_type:
            break
    print("Done.")    
    

if __name__ == "__main__":
    env, agent, task, planner, viz = setup()
    solve(env, agent, task, planner, viz)    
