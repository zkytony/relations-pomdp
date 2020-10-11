from relpomdp.home2d.tasks.search_item.search_item_task import *
from relpomdp.home2d.tasks.search_item.visual import SearchItemViz
from relpomdp.home2d.domain.maps import all_maps
import relpomdp.oopomdp.framework as oopomdp
from relpomdp.oopomdp.framework import Objstate, Objobs, OOObs
import time

def setup():
    grid_map = all_maps["map_small_0"]()

    # Building object state
    robot_id = 1
    robot_state = Objstate("Robot",
                           pose=(0,0,0),
                           camera_direction="+x")
    salt_id = 10
    salt_state = Objstate("Salt",
                          pose=(3,3))

    pepper_id = 15
    pepper_state = Objstate("Pepper",
                            pose=(3,2))

    # Make task
    sensor = Laser2DSensor(robot_id, grid_map,  # the sensor uses the grid map for wall blocking
                           fov=90, min_range=1, max_range=2,
                           angle_increment=0.5)
    task = SearchItemTask(robot_id, salt_id, "Salt", sensor, grid_map=grid_map)
    
    # Make environment
    init_state = {robot_id: robot_state,
                  salt_id: salt_state,
                  pepper_id: pepper_state}
    env = task.get_env(init_state=init_state,
                       grid_map=grid_map)

    target_id = salt_id
    target_class = task.target_class
    salt_state["is_found"] = False

    # Obtain prior
    prior_type = "uniform"
    init_belief = pomdp_py.OOBelief({target_id: task.get_prior(grid_map,
                                                               prior_type=prior_type,
                                                               env=env),
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
        action, observation, reward = task.step(env, agent, planner)
        print("     action: %s" % str(action.name))        
        print("     reward: %s" % str(reward))
        print("observation: %s" % str(observation))

        time.sleep(1)
        
        viz.update(agent.belief)
        viz.on_loop()
        viz.on_render()

        if task.is_done(env):
            break
    print("Done.")    

if __name__ == "__main__":
    env, agent, task, planner, viz = setup()
    solve(env, agent, task, planner, viz)
