from relpomdp.home2d.tasks.search_room.search_room_task import *
from relpomdp.home2d.tasks.search_room.visual import SearchRoomViz
from relpomdp.home2d.domain.maps import all_maps
import relpomdp.oopomdp.framework as oopomdp
import time


def setup():
    grid_map = all_maps["map_small_1"]()

    # Building object state
    robot_id = 1
    robot_state = objstate("Robot",
                           pose=(0,0,0),
                           camera_direction="+x")

    room_type = "Office"  # also used as id

    # Make task
    task = SearchRoomTask(robot_id, room_type, grid_map=grid_map)

    room = None
    for room_name in grid_map.rooms:
        if grid_map.rooms[room_name].room_type == room_type:
            room = grid_map.rooms[room_name]
            break
    
    # Make environment
    init_state = {robot_id: robot_state,
                  room_type: objstate(room_type, pose=room.center_of_mass, reached=False)}
    env = Home2DEnvironment(robot_id,
                            grid_map, init_state,
                            reward_model=task.reward_model)
    env.transition_model.cond_effects.append(
        (CanStop(), StopEffect(robot_id, room_type, grid_map))
    )
    cond_effects_o = [(CanObserve(),
                       RoomObserveEffect(robot_id, room_type, epsilon=1e-12,
                                         grid_map=grid_map,
                                         for_env=True))]
    env.observation_model = oopomdp.OOObservationModel(cond_effects_o)

    # Obtain prior
    prior_type = "uniform"
    init_belief = pomdp_py.OOBelief({room_type: task.get_prior(grid_map, prior_type=prior_type),
                                     robot_id: pomdp_py.Histogram({env.robot_state:1.0})})
    agent = task.to_agent(init_belief)

    # Create planner and make a plan
    planner = pomdp_py.POUCT(max_depth=20,
                             discount_factor=0.95,
                             num_sims=500,
                             exploration_const=200,
                             rollout_policy=agent.policy_model)

    viz = SearchRoomViz(env,
                        {room_type: (100, 200, 20)},
                        res=30)
    
    return env, agent, task, planner, viz


def solve(env, agent, task, planner, viz):
    viz.on_init()
    viz.update(agent.belief)    
    viz.on_render()

    for step in range(100):
        print("---- Step %d ----" % step)
        action = planner.plan(agent)
        for a in agent.tree.children:
            print(a, agent.tree.children[a].value)
        
        reward = env.state_transition(action, execute=True)
        observation = env.observation_model.sample(env.state, action)

        # Belief update
        robot_state = env.state.object_states[task.robot_id]
        room_types = set(env.grid_map.rooms[r].room_type
                         for r in env.grid_map.rooms)
        # Create a next state space on top of new robot pose, and all room types.
        cur_belief = pomdp_py.Histogram({
            oopomdp.OOState({task.room_type: room_state,
                             task.robot_id: robot_state}) : agent.belief.object_beliefs[task.room_type][room_state]
            for room_state in agent.belief.object_beliefs[task.room_type]})        
        new_belief = pomdp_py.update_histogram_belief(cur_belief,
                                                      action, observation,
                                                      agent.observation_model,
                                                      agent.transition_model,
                                                      static_transition=True)
        # Take just the room state from this
        new_belief = pomdp_py.Histogram({state.object_states[task.room_type]:
                                         new_belief[state]
                                         for state in new_belief})
        agent.belief.set_object_belief(task.room_type, new_belief)        
        agent.belief.set_object_belief(task.robot_id, pomdp_py.Histogram({robot_state:1.0}))
        observation_planner = agent.observation_model.sample(env.state, action)
        planner.update(agent, action, observation_planner)
        
        print("     action: %s" % str(action.name))        
        print("     reward: %s" % str(reward))
        print("observation: %s" % str(observation))
        print("robot state: %s" % str(robot_state))

        time.sleep(1)
        
        viz.on_loop()
        viz.update(agent.belief)        
        viz.on_render()

        if task.is_done(env, action):
            break
    print("Done.")    
    

if __name__ == "__main__":
    env, agent, task, planner, viz = setup()
    solve(env, agent, task, planner, viz)    
