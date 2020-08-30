from relpomdp.object_search.world_specs.build_world import *
from relpomdp.object_search.env import *
from relpomdp.object_search.sensor import *
from relpomdp.object_search.agent import *
from relpomdp.object_search.greedy_planner import GreedyPlanner, RandomPlanner
from relpomdp.pgm.mrf import SemanticMRF, relations_to_mrf
from relpomdp.object_search.relation import *
from relpomdp.object_search.tests.result_types import *
from relpomdp.object_search.tests.result_types import *
from relpomdp.object_search.tests.worlds import *
import networkx as nx
import matplotlib.pyplot as plt
from pgmpy.inference import BeliefPropagation
import pomdp_py
import pygame
import time
import copy
from sciex import Trial, Event

class SingleObjectSearchTrial(Trial):
    """Single object search"""

    RESULT_TYPES = [HistoryResult]

    def __init__(self, name, config, verbose=False):
        super().__init__(name, config, verbose=verbose)

    def run(self, logging=False):
        world = self._config["world"]
        print("Creating map ...")
        ids, grid_map, init_state, relations, colors = world(**self._config["world_configs"])

        print("Creating environment ...")
        env = ObjectSearchEnvironment(ids,
                                      grid_map,
                                      init_state)
        mrf = relations_to_mrf(relations)

        target_variable = self._config["target_variable"]
        target_class = target_variable.split("_")[0]
        target_phi = mrf.query(variables=[target_variable])
        target_phi.normalize()
        target_hist_mrf = {}
        for loc in mrf.values(target_variable):
            state = ItemState(target_class, loc)
            target_hist_mrf[state] = target_phi.get_value({target_variable:loc})

        target_id = ids["Target"][0]
        robot_id = ids["Robot"]
        init_belief = pomdp_py.OOBelief({target_id: pomdp_py.Histogram(target_hist_mrf),
                                         robot_id: pomdp_py.Histogram({env.robot_state:1.0})})
        sensor = Laser2DSensor(robot_id, env.grid_map, fov=90, min_range=1, max_range=2,
                               angle_increment=0.5)
        agent = ObjectSearchAgent(env.grid_map, sensor, env.ids,
                                  init_belief)

        viz = None
        if self._config["visualize"]:
            print("Creating visualization ...")
            objcolors = {}
            for objid in env.state.object_states:
                s_o = env.state.object_states[objid]
                if s_o.objclass in colors:
                    objcolors[objid] = colors[s_o.objclass]
            viz = ObjectSearchViz(env,
                                  objcolors,
                                  res=30,
                                  controllable=True,
                                  img_path="../imgs")
            viz.on_init()
            viz.on_render()    
            viz.update({target_id:target_hist_mrf})
            viz.on_render()        

        if self._config["planner_type"] == "pouct":
            planner = pomdp_py.POUCT(max_depth=self._config["planner"]["max_depth"],
                                     discount_factor=self._config["planner"]["discount_factor"],
                                     num_sims=self._config["planner"]["num_sims"],
                                     exploration_const=self._config["planner"]["exploration_const"],
                                     rollout_policy=agent.policy_model)
            
        elif self._config["planner_type"] == "greedy":
            planner = GreedyPlanner(ids)
            
        elif self._config["planner_type"] == "random":
            planner = RandomPlanner(ids)

        _History = [(copy.deepcopy(env.state),None,None,0)]  # s,a,o,r

        used_objects = set()  # objects who has contributed to mrf belief update
        for step in range(self._config["max_steps"]):
            print("---- Step %d ----" % step)                    
            # Input action from keyboard
            if self._config["visualize"] and self._config["user_control"]:
                action = None
                while action is None:
                    for event in pygame.event.get():
                        if event.type == pygame.KEYDOWN:
                            action = viz.on_event(event)
                    time.sleep(0.1)
            else:
                action = planner.plan(agent)

                if isinstance(planner, pomdp_py.POUCT):
                    print("   num sims: %d" % planner.last_num_sims)
                else:
                    time.sleep(0.1)

            robot_state = env.robot_state.copy()
            reward = env.state_transition(action, execute=True)
            print("robot state: %s" % str(env.robot_state))
            print("     action: %s" % str(action.name))
            print("     reward: %s" % str(reward))

            observation = agent.observation_model.sample(env.state, action)
            print("observation: %s" % str(observation))

            # Compute a distribution using the MRF
            updating_mrf = False
            for objid in observation.object_observations:
                o_obj = observation.object_observations[objid]
                if objid != env.ids["Robot"] and objid not in used_objects:
                    if o_obj.objclass == target_class:
                        # You just observed the target. MRF isn't useful here.
                        continue

                    target_phi = mrf.query(variables=[target_variable],
                                         evidence=o_obj.to_evidence())
                    target_phi.normalize()
                    for loc in mrf.values(target_variable):
                        state = ItemState(target_class, loc)
                        target_hist_mrf[state] = target_phi.get_value({target_variable:loc})
                    updating_mrf = True
                    used_objects.add(objid)

            # Compute a distribution using the standard belief update
            current_target_hist = agent.belief.object_beliefs[target_id]
            next_robot_state = env.robot_state.copy()

            new_histogram = {}  # state space still the same.
            total_prob = 0
            for next_target_state in current_target_hist:
                next_state = JointState({target_id: next_target_state,
                                         robot_id: next_robot_state})
                observation_prob = agent.observation_model.probability(
                    observation.for_objs([robot_id,target_id]), next_state, action)
                mrf_prob = 1.0
                if updating_mrf:
                    mrf_prob = target_hist_mrf[next_target_state]

                transition_prob = current_target_hist[next_target_state]
                new_histogram[next_target_state] = mrf_prob * observation_prob * transition_prob
                total_prob += new_histogram[next_target_state]

            # Normalize
            for target_state in new_histogram:
                if total_prob > 0:
                    new_histogram[target_state] /= total_prob
            target_hist_update = new_histogram

            agent.set_belief(pomdp_py.OOBelief({target_id:pomdp_py.Histogram(target_hist_update),
                                                robot_id:pomdp_py.Histogram({env.robot_state:1.0})}))
            planner.update(agent, action, observation)

            if viz is not None:
                viz.update({target_id: target_hist_update})
                viz.on_loop()
                viz.on_render()

            # Record result
            _History.append((copy.deepcopy(env.state), action, observation, reward))

            # Terminates
            if env.state.object_states[target_id].is_found:
                break

        if viz is not None:
            viz.on_cleanup()
            
        return [HistoryResult(_History)]


if __name__ == "__main__":
    config = {
        "world": salt_pepper_1,
        "world_configs": {},
        "target_variable": "Salt_pose",
        "planner_type": "pouct",
        "planner": {
            "max_depth": 20,
            "discount_factor": 0.95,
            "num_sims": 200,
            "exploration_const": 100
        },
        "max_steps": 100,
        "visualize": True,
        "user_control": False
    }
    trial = SingleObjectSearchTrial("salt_pepper",
                                    config, verbose=True)
    trial.run()
    
