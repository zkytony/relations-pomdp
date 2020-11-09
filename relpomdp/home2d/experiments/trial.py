from sciex import Experiment, Trial, Event, Result,\
    YamlResult, PklResult, PostProcessingResult
import pickle
import os
import copy



class RelPOMDPTrial(Trial):
    RESULT_TYPES = [RewardsResult, StatesResult]
    def __init__(self, name, config, verbose=False):
        super().__init__(name, config, verbose=verbose)

    def logg(self, info):
        self.log_event(Event("Trial {} | {}".format(self.name, info)))

    def run(self, logging=False):
        # Path where the test environments are stored
        env_path = self._config["env_path"]

        # ID of the environment to be used for this trial
        env_id = self._config["env_id"]

        with open(env_path, "rb") as f:
            env = pickle.load(f)[env_id]

        # Domain config; These determine how an agent is instantiated
        domain_config = self._config["domain"]

        # Planning config: Parameters for the pomdp planning algorithm
        planning_config = self._config["planning"]

        # Agent type
        agent_type = self._config["agent_type"].lower()
        assert agent_type in {"mdp",             # knows where target is
                              "pomdp",           # only cares about the target, in full map
                              "pomdp-subgoal",   # uses subgoal, with full map
                              "pomdp-nk",        # only cares about the target, no full map
                              "pomdp-subgoal-nk",   # uses subgoal, no full map
                              "pomdp-subgoal-nk-no_corr",   # uses subgoal, no correlation belief udpate, no full map
                              "random-nk"}       # randomly move to unexplored place, no full map

        # Target class
        target_class = self._config["target_class"]

        # target sensor and slam sensor params
        target_sensor_config = {}
        slam_sensor_config = {}
        for sensor_name in config["sensors"]:
            cfg = config["sensors"][sensor_name]
            if target_class in cfg["noises"]:
                target_sensor_config = copy.deepcopy(cfg)
                target_sensor_config["noises"] = target_sensor_config["noises"][target_class]
            # Use room sensor as slam sensor
            if sensor_name.lower().startswith("room"):
                slam_sensor_config = copy.deepcopy(cfg)

        # Load scoring files, for subgoal agents
        if "subgoal" in agent_type:
            df_corr = pd.read_csv(self._config["corr_score_file"])
            df_dffc = pd.read_csv(self._config["diffc_score_file"])
            df_subgoal = pd.read_csv(self._config["args.subgoal_score_file"])

        # Run
        visualize = self._config["visualize"]
        if agent_type == "mdp":
            rewards, states, history = test_mdp(env, target_class,
                                                target_sensor_config=target_sensor_config,
                                                slam_sensor_config=slam_sensor_config,
                                                visualize=visualize,
                                                logger=self.logg,
                                                **planning_config)
        elif agent_type == "pomdp":
            rewards, states, history = test_pomdp(env, target_class,
                                                  target_sensor_config=target_sensor_config,
                                                  slam_sensor_config=slam_sensor_config,
                                                  visualize=visualize,
                                                  logger=self.logg,
                                                  **params)
        elif agent_type == "pomdp-nk":
            rewards, states, history = test_pomdp_nk(env, target_class,
                                                     target_sensor_config=target_sensor_config,
                                                     slam_sensor_config=slam_sensor_config,
                                                     visualize=visualize,
                                                     logger=self.logg,
                                                     **params)
        elif agent_type == "pomdp-subgoal":
            rewards, states, history = test_subgoals_agent(env_copy, args.target_class, config,
                                                           df_corr, df_dffc, df_subgoal,
                                                           use_correlation_belief_update=True,
                                                           visualize=visualize,
                                                           full_map=True,
                                                           logger=self.logg,
                                                           **params)
        elif agent_type == "pomdp-subgoal-nk":
            rewards, states, history = test_subgoals_agent(env_copy, args.target_class, config,
                                                           df_corr, df_dffc, df_subgoal,
                                                           use_correlation_belief_update=True,
                                                           visualize=visualize,
                                                           full_map=False,
                                                           logger=self.logg,
                                                           **params)
        elif agent_type == "pomdp-subgoal-nk-no_corr":
            rewards, states, history = test_subgoals_agent(env_copy, args.target_class, config,
                                                           df_corr, df_dffc, df_subgoal,
                                                           use_correlation_belief_update=False,
                                                           visualize=visualize,
                                                           full_map=False,
                                                           logger=self.logg,
                                                           **params)
        elif agent_type == "random-nk":
            rewards, states, history = test_random_nk(env_copy, args.target_class,
                                                      slam_sensor_config=slam_sensor_config,
                                                      visualize=visualize,
                                                      logger=self.logg,
                                                      **params)
        else:
            if logging:
                self.log_event(Event("Trial %s | ERROR unknown agent type %s" % (self.name, agent_type)))

        results = [RewardsResult(rewards),
                   StatesResult(states),
                   HistoryResult(history)]
        return results
