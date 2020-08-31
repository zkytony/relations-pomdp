import argparse
import os
import pickle
import yaml
import re
import time
import pomdp_py
import subprocess
import copy
from relpomdp.object_search.tests.worlds import *

ABS_PATH = os.path.dirname(os.path.abspath(__file__))

def main():
    parser = argparse.ArgumentParser(description="replay a trial")
    parser.add_argument("trial_path", type=str, help="Path to trial directory")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    with open(os.path.join(args.trial_path, "history.pkl"), "rb") as f:
        history = pickle.load(f)

    with open(os.path.join(args.trial_path, "trial.pkl"), "rb") as f:
        trial = pickle.load(f)

    trial.config["visualize"] = True
    trial.config["img_path"] = os.path.join(ABS_PATH, "../imgs")
    env, agent, planner, mrf, viz = trial.setup()
    target_id = env.ids["Target"][0]
    robot_id = env.ids["Robot"]    

    # Visualization
    viz.on_init()
    viz.update({target_id:agent.belief.object_beliefs[target_id]})
    viz.on_render()

    # TODO: You should save the prior as a pkl file
    used_objects = set()  # objects who has contributed to mrf belief update
    discount = 1.0
    discounted_reward = 0.0    
    for i in range(len(history)):
        print("---- Step %d ----" % i)        
        state, action, _, reward = history[i]
        observation = agent.observation_model.sample(state, action)
        next_robot_state = copy.deepcopy(env.robot_state)
        env.apply_transition(state)
        trial.belief_update(agent, action, observation, next_robot_state, mrf, used_objects)

        discounted_reward += discount * reward
        discount *= trial.config["planner"]["discount_factor"]        
        print("robot state: %s" % str(env.robot_state))
        print("     reward: %s" % str(reward))
        print("disc reward: %.3f" % discounted_reward)
        print("observation: %s" % str(observation))
        
        viz.update({target_id: agent.belief.object_beliefs[target_id]})
        time.sleep(0.1)
        viz.on_loop()        
        viz.on_render()
    viz.on_cleanup()        
        
        
if __name__ == "__main__":
    main()
