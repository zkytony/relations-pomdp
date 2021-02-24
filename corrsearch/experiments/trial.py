import pomdp_py
from sciex import Experiment, Trial, Event, Result,\
    YamlResult, PklResult, PostProcessingResult
from corrsearch.models import *
from corrsearch.objects import *
from corrsearch.experiments.result_types import *
import corrsearch
import time

class SearchTrial(Trial):

    RESULT_TYPES = [RewardsResult, StatesResult, HistoryResult]

    def __init__(self, name, config, verbose=False):
        """
        Args:
            config (dict): Configuration
                "problem_creator": method that returns SearchProblem,
                "problem_config": dict(),   configuration to create the problem
                "instance_config": dict(),  configuration to create an instance
                "viualize": bool,           True if visualization is on
                "visualize_config": dict(), configuration for visualization
                "exec_config": dict9),      configuration when simulating the agent
                "planner": str or class     planner to use,
                "planner_init_config": dict(),     config to initialize planner
                "planner_exec_config": dict(),     config to pass in to planner.plan

        """
        super().__init__(name, config, verbose=verbose)


    def _do_info(self, step, action, observation, reward, cum_reward):
        _step_info = "Step {}:   Action: {}   Reward: {}    Cumulative Reward: {}\n"\
                     .format(step, action, reward, cum_reward)
        for objid in observation:
            zi = observation[objid]
            if isinstance(zi, NullObz):
                _step_info += "    z(%d) = Null\n" % objid
            else:
                _step_info += "    z({}) = {}\n".format(objid, zi)
        _step_info += "\n"
        return _step_info


    def run(self, logging=False):
        for import_cmd in self.config.get("imports", []):
            exec(import_cmd)
        problem_creator = eval(self.config["problem_creator"])
        problem = problem_creator(**self.config["problem_config"])
        env, agent = problem.instantiate(**self.config["instance_config"])
        if self.config["planner"].endswith("sarsop"):
            planner = pomdp_py.sarsop(agent, **self.config["planner_init_config"])
        elif self.config["planner"].lower().endswith("pouct"):
            planner = eval(self.config["planner"])(rollout_policy=agent.policy_model,
                                                   **self.config["planner_init_config"])
        else:
            planner = eval(self.config["planner"])(**self.config["planner_init_config"])

        if self.config["visualize"]:
            viz = problem.visualizer(**self.config["visualize_config"])
            viz.visualize(env.state, belief=agent.belief)

        _History = []
        _Rewards = []
        _States = [copy.deepcopy(env.state)]
        _cum_reward = 0
        _discount = 1.0

        max_steps = self.config["exec_config"].get("max_steps", 100)
        for step in range(max_steps):
            import pdb; pdb.set_trace()
            if self.config["planner"] == "PlaybackPlanner":
                action, observation = planner.plan(agent, **self.config["planner_exec_config"])
                reward = env.state_transition(action, execute=True)
            else:
                action = planner.plan(agent, **self.config["planner_exec_config"])
                reward = env.state_transition(action, execute=True)
                observation = env.provide_observation(agent.observation_model, action)
                if self.config["exec_config"].get("debugging", False):
                    agent.tree.print_children_value()

            agent.set_belief(agent.belief.update(agent, observation, action))
            planner.update(agent, action, observation)

            # Info and logging
            _cum_reward += reward * _discount
            _discount *= self.config["discount_factor"]
            _step_info = self._do_info(step, action, observation,
                                       reward, _cum_reward)
            if logging:
                self.log_event(Event("Trial {} | {}".format(self.name, _step_info)))
            else:
                print(_step_info)

            _History.append((action, observation))
            _States.append(copy.deepcopy(env.state))
            _Rewards.append(reward)

            if self.config["visualize"]:
                viz.visualize(env.state, belief=agent.belief)

            if isinstance(action, Declare):
                if logging:
                    self.log_event(Event("Trial {} | Declared. Done.".format(self.name)))
                else:
                    print("Declared. Done.")
                break

            time.sleep(self.config["exec_config"].get("step_delay", 0.1))

        results = [
            RewardsResult(_Rewards),
            StatesResult(_States),
            HistoryResult(_History)
        ]
        return results
