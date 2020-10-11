from relpomdp.home2d.tasks.leave_room.leave_room_task import *
from relpomdp.home2d.domain.visual import Home2DViz
from relpomdp.home2d.domain.maps import all_maps
from relpomdp.home2d.domain.env import Home2DEnvironment
import relpomdp.oopomdp.framework as oopomdp
from relpomdp.oopomdp.framework import Objstate, Objobs, OOObs
import time

def setup():
    grid_map = all_maps["map_small_1"]()

    # Building object state
    robot_id = 1
    robot_state = Objstate("Robot",
                           pose=(0,0,0),
                           camera_direction="+x",
                           room_id=grid_map.room_of((0,0)).name)
    room_id = robot_state["room_id"]
    
    task = LeaveRoomTask(robot_id, room_id, grid_map=grid_map)
    
    # Make environment
    init_state = {robot_id: robot_state}
    env = Home2DEnvironment(robot_id,
                            grid_map, init_state, reward_model=task.reward_model)
    cond_effects_t = []
    cond_effects_t.append((CanMove(robot_id, env.legal_motions),
                           MoveEffectWithRoom(robot_id, grid_map)))
    env.transition_model._cond_effects = cond_effects_t
        

    init_belief = pomdp_py.OOBelief({robot_id: pomdp_py.Histogram({env.robot_state:1.0})})
    # Make agent
    agent = task.to_agent(init_belief)

    # Create planner and make a plan
    planner = pomdp_py.POUCT(max_depth=20,
                             discount_factor=0.95,
                             num_sims=300,
                             exploration_const=100,
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
        agent.belief.set_object_belief(task.robot_id, pomdp_py.Histogram({env.robot_state:1.0}))
        planner.update(agent, action, observation)
        
        print("     action: %s" % str(action.name))        
        print("     reward: %s" % str(reward))
        print("observation: %s" % str(observation))
        print("robot state: %s" % str(env.robot_state))

        time.sleep(1)
        
        viz.on_loop()
        viz.on_render()

        print(env.robot_state["room_id"])        
        if env.robot_state["room_id"] != task.room_id:
            break
    print("Done.")    

if __name__ == "__main__":
    env, agent, task, planner, viz = setup()
    solve(env, agent, task, planner, viz)
