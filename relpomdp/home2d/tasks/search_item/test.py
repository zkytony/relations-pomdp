from relpomdp.home2d.tasks.search_item.search_item_task import *
from relpomdp.home2d.tasks.search_item.visual import SearchItemViz
from relpomdp.home2d.domain.maps import all_maps
import relpomdp.oopomdp.framework as oopomdp
import time

def setup():
    grid_map = all_maps["map_small_0"]()

    # Building object state
    robot_id = 1
    robot_state = objstate("Robot",
                           pose=(0,0,0),
                           camera_direction="+x")
    salt_id = 10
    salt_state = objstate("Salt",
                          pose=(3,3))

    pepper_id = 15
    pepper_state = objstate("Pepper",
                            pose=(3,2))

    # Make task
    sensor = Laser2DSensor(robot_id, grid_map,  # the sensor uses the grid map for wall blocking
                           fov=90, min_range=1, max_range=2,
                           angle_increment=0.5)
    task = SearchItemTask(robot_id, salt_id, sensor, grid_map=grid_map)
    
    # Make environment
    init_state = {robot_id: robot_state,
                  salt_id: salt_state,
                  pepper_id: pepper_state}
    env = Home2DEnvironment(robot_id,
                            grid_map, init_state, reward_model=task.reward_model)

    target_id = salt_id
    target_class = "Salt"
    salt_state["is_found"] = False

    # Obtain prior
    prior_type = "uniform"
    target_hist = {}
    total_prob = 0
    for x in range(grid_map.width):
        for y in range(grid_map.length):
            state = objstate(target_class, pose=(x,y), is_found=False)
            if prior_type == "uniform":
                target_hist[state] = 1.0
            elif prior_type == "informed":
                if (x,y) != env.state.object_states[target_id]["pose"]:
                    target_hist[state] = 0.0
                else:
                    target_hist[state] = 1.0
            total_prob += target_hist[state]
    # Normalize
    for state in target_hist:
        target_hist[state] /= total_prob

    init_belief = pomdp_py.OOBelief({target_id: pomdp_py.Histogram(target_hist),
                                     robot_id: pomdp_py.Histogram({env.robot_state:1.0})})
    # Make agent
    agent = task.to_agent(init_belief)
    env.transition_model.cond_effects.append((CanPickup(task.robot_id, task.target_id),
                                              PickupEffect()))
    

    # Create planner and make a plan
    planner = pomdp_py.POUCT(max_depth=10,
                             discount_factor=0.95,
                             num_sims=100,
                             exploration_const=100,
                             rollout_policy=agent.policy_model)

    viz = SearchItemViz(env,
                        {salt_id: (128, 128, 128),
                         pepper_id: (200, 10, 10)},
                        res=30)
    
    return env, agent, task, planner, viz


def solve(env, agent, task, planner, viz):
    viz.on_init()
    viz.update(agent.belief)    
    viz.on_render()
    
    for step in range(100):
        print("---- Step %d ----" % step)
        action = planner.plan(agent)
        reward = env.state_transition(action, execute=True)
        observation = agent.observation_model.sample(env.state, action)

        # Belief update
        robot_state = env.state.object_states[task.robot_id]
    
        cur_belief = pomdp_py.Histogram({
            oopomdp.OOState({task.target_id: target_state,
                             task.robot_id: robot_state}) : agent.belief.object_beliefs[task.target_id][target_state]
            for target_state in agent.belief.object_beliefs[task.target_id]})
        new_belief = pomdp_py.update_histogram_belief(cur_belief,
                                                      action, observation,
                                                      agent.observation_model,
                                                      agent.transition_model,
                                                      static_transition=True)
        # Take just the target state from this
        new_belief = pomdp_py.Histogram({state.object_states[task.target_id]:
                                         new_belief[state]
                                         for state in new_belief})
        agent.belief.set_object_belief(task.target_id, new_belief)
        agent.belief.set_object_belief(task.robot_id, pomdp_py.Histogram({robot_state:1.0}))
        planner.update(agent, action, observation)
        
        print("     action: %s" % str(action.name))        
        print("     reward: %s" % str(reward))
        print("observation: %s" % str(observation))
        print("robot state: %s" % str(robot_state))

        time.sleep(1)
        
        viz.update(agent.belief)
        viz.on_loop()
        viz.on_render()

        if env.state.object_states[task.target_id]["is_found"]:
            break
    print("Done.")    

if __name__ == "__main__":
    env, agent, task, planner, viz = setup()
    solve(env, agent, task, planner, viz)
