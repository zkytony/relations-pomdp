from relpomdp.object_search.world_specs.build_world import *
from relpomdp.object_search.env import *
from relpomdp.object_search.sensor import *
from relpomdp.object_search.agent import *
from relpomdp.object_search.greedy_planner import GreedyPlanner, RandomPlanner
from relpomdp.object_search.subgoal_planner import SubgoalPlanner, ReachRoomSubgoal
from relpomdp.object_search.utils import save_images_and_compress
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
import subprocess
from sciex import Trial, Event

class SingleObjectSearchTrial(Trial):
    """Single object search"""

    RESULT_TYPES = [HistoryResult]

    def __init__(self, name, config, verbose=False):
        super().__init__(name, config, verbose=verbose)


    def setup(self):
        world = self._config["world"]
        print("Creating map ...")
        ids, grid_map, init_state, mrf, colors = world(**self._config["world_configs"])
        
        target_id = ids["Target"][0]
        robot_id = ids["Robot"]

        print("Creating environment ...")
        env = ObjectSearchEnvironment(ids,
                                      grid_map,
                                      init_state)
        
        target_variable = self._config["target_variable"]
        target_class = target_variable.split("_")[0]

        if self._config["prior_type"] == "mrf":
            target_phi = mrf.query(variables=[target_variable], verbose=True)

        # Obtain prior
        target_hist = {}
        total_prob = 0
        for x in range(grid_map.width):
            for y in range(grid_map.length):
                state = ItemState(target_class, (x,y))
                if self._config["prior_type"] == "uniform":
                    target_hist[state] = 1.0
                elif self._config["prior_type"] == "mrf":
                    target_hist[state] = target_phi.get_value({target_variable:(x,y)})
                elif self._config["prior_type"] == "informed":
                    if (x,y) != env.state.object_states[target_id].pose:
                        target_hist[state] = 0.0
                    else:
                        target_hist[state] = 1.0
                total_prob += target_hist[state]
        # Normalize
        for state in target_hist:
            target_hist[state] /= total_prob
            
        init_belief = pomdp_py.OOBelief({target_id: pomdp_py.Histogram(target_hist),
                                         robot_id: pomdp_py.Histogram({env.robot_state:1.0})})
        sensor = Laser2DSensor(robot_id, env.grid_map, fov=90, min_range=1, max_range=2,
                               angle_increment=0.5)
        agent = ObjectSearchAgent(env.grid_map, sensor, env.ids,
                                  init_belief)

        # Create planner
        if self._config["planner_type"].startswith("pouct"):
            planner = pomdp_py.POUCT(max_depth=self._config["planner"]["max_depth"],
                                     discount_factor=self._config["planner"]["discount_factor"],
                                     num_sims=self._config["planner"]["num_sims"],
                                     exploration_const=self._config["planner"]["exploration_const"],
                                     rollout_policy=agent.policy_model,
                                     action_prior=agent.policy_model.action_prior)
            
        elif self._config["planner_type"].startswith("greedy"):
            planner = GreedyPlanner(ids)
            
        elif self._config["planner_type"].startswith("random"):
            planner = RandomPlanner(ids)

        if self._config["planner_type"].endswith("subgoal"):
            subgoals = {}
            for subgoal_str in self._config["planner"]["subgoals"]:
                # TODO: More types of subgoals?
                sg = ReachRoomSubgoal(env.ids, subgoal_str, env.grid_map)
                subgoals[sg.name] = sg
            planner = SubgoalPlanner(env.ids, subgoals,
                                     env.grid_map, planner)

        # Visualization
        viz = None
        game_states = []
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
                                  img_path=self._config["img_path"])
            viz.on_init()
            viz.on_render()
            viz.update({target_id:target_hist})
            img = viz.on_render()
            game_states.append(img)

        return env, agent, planner, mrf, viz, game_states

    def run(self, logging=False):
        env, agent, planner, mrf, viz, game_states = self.setup()
        target_id = env.ids["Target"][0]
        robot_id = env.ids["Robot"]        
        
        _History = [(copy.deepcopy(env.state),None,None,0)]  # s,a,o,r
        used_cues = set()  # objects who has contributed to mrf belief update
        discount = 1.0
        discounted_reward = 0
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

                if hasattr(planner, "last_num_sims"):
                    print("   num sims: %d" % planner.last_num_sims)
                else:
                    time.sleep(0.1)

            reward = env.state_transition(action, execute=True)
            observation = agent.observation_model.sample(env.state, action)
            next_robot_state = env.robot_state.copy()
            self.belief_update(agent, action, observation, next_robot_state, mrf, used_cues)
            planner.update(agent, action, observation)

            discounted_reward += discount * reward
            discount *= self.config["planner"]["discount_factor"]
            print("robot state: %s" % str(env.robot_state))
            print("     action: %s" % str(action.name))
            print("     reward: %s" % str(reward))
            print("disc reward: %.3f" % discounted_reward)
            print("observation: %s" % str(observation))
            
            if viz is not None:
                viz.update({target_id: agent.belief.object_beliefs[target_id]})
                viz.on_loop()
                img = viz.on_render()
                game_states.append(img)
            
            # Record result
            _History.append((copy.deepcopy(env.state),
                             copy.deepcopy(action),
                             copy.deepcopy(observation), reward))

            # Terminates
            if env.state.object_states[target_id].is_found:
                break

        save_game_states = self.config.get("save_game_states", False)
        game_states_save_dir = self.config.get("game_states_save_dir", "./")
        if save_game_states:
            print("Saving images...")
            save_images_and_compress(game_states,
                                     game_states_save_dir)
            subprocess.Popen(["nautilus", game_states_save_dir])
            
        if viz is not None:
            viz.on_cleanup()
        return [HistoryResult(_History)]


    def belief_update(self, agent, action, observation, next_robot_state, mrf, used_cues):
        # Compute a distribution using the MRF
        target_id = agent.ids["Target"][0]
        robot_id = agent.ids["Robot"]
        target_variable = self._config["target_variable"]
        target_class = target_variable.split("_")[0]        
        if self._config["using_mrf_belief_update"]:
            updating_mrf = False
            target_hist_mrf = {}
            observations = [observation]
            if isinstance(observation, CombinedObservation):
                observations = observation.observations

            evidence = {}
            for o in observations:
                if isinstance(o, JointObservation):
                    for objid in o.object_observations:
                        o_obj = o.object_observations[objid]
                        if objid != robot_id and objid not in used_cues:
                            if o_obj.objclass == target_class:
                                # You just observed the target. MRF isn't useful here.
                                continue
                            if mrf.valid_var("%s_pose" % o_obj.objclass):
                                evidence.update(o_obj.to_evidence())
                                used_cues.add(objid)
                elif isinstance(o, RoomObservation):
                    if mrf.valid_var(o.room_type)\
                       and o.name not in used_cues:
                        evidence.update(o.to_evidence())
                        used_cues.add(o.name)
                        
            if len(evidence) > 0:
                target_phi = mrf.query(variables=[target_variable],
                                       evidence=evidence, verbose=True)
                target_phi.normalize()
                for loc in mrf.values(target_variable):
                    state = ItemState(target_class, loc)
                    target_hist_mrf[state] = target_phi.get_value({target_variable:loc})
                updating_mrf = True

        # Compute a distribution using the standard belief update
        current_target_hist = agent.belief.object_beliefs[target_id]

        new_histogram = {}  # state space still the same.
        total_prob = 0
        for next_target_state in current_target_hist:
            next_state = JointState({target_id: next_target_state,
                                     robot_id: next_robot_state})
            observation_prob = agent.observation_model.probability(
                observation, next_state, action)
            mrf_prob = 1.0
            if self._config["using_mrf_belief_update"] and updating_mrf:
                mrf_prob = target_hist_mrf[next_target_state]

            transition_prob = current_target_hist[next_target_state]
            new_histogram[next_target_state] = mrf_prob * observation_prob * transition_prob
            if next_target_state.pose == (0,0):
                print(mrf_prob, observation_prob, transition_prob)
            total_prob += new_histogram[next_target_state]

        # Normalize
        for target_state in new_histogram:
            if total_prob > 0:
                new_histogram[target_state] /= total_prob
        target_hist_update = new_histogram
        agent.set_belief(pomdp_py.OOBelief({target_id:pomdp_py.Histogram(target_hist_update),
                                            robot_id:pomdp_py.Histogram({next_robot_state:1.0})}))



if __name__ == "__main__":
    config = {
        "world": salt_pepper_1,
        "world_configs": {},
        "target_variable": "Salt_pose",
        "planner_type": "pouct-subgoal",
        "planner": {
            "max_depth": 20,
            "discount_factor": 0.95,
            "num_sims": 500,
            "exploration_const": 200,
            "subgoals": ["Kitchen"]
        },
        "prior_type": "mrf",
        "using_mrf_belief_update": True,
        "max_steps": 100,
        "visualize": True,
        "user_control": False,
        "img_path": "../imgs",
        "save_game_states": True,
        "game_states_save_dir": "saved_trials",
    }
    trial = SingleObjectSearchTrial("salt_pepper",
                                    config, verbose=True)
    trial.run()
    
