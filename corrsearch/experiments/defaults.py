POMCP_PLANNER_CONFIG = {
    "max_depth": 25,
    "discount_factor": 0.95,
    "num_sims": 1000,
    "exploration_const": 200
}
ENTROPY_PLANNER_CONFIG = {
    "declare_threshold": 0.9,
    "entropy_improvement_threshold": 1e-3,
    "num_samples": 100
}
RANDOM_PLANNER_CONFIG = {
    "declare_threshold": 0.9,
}
HEURISTIC_ONLINE_PLANNER_CONFIG = {
    **POMCP_PLANNER_CONFIG,
    "k": 2,
    "num_zsamples": 30,
    "num_rsamples": 30,
    "gamma": 0.95,
    "num_visits_init": 0,
    "init_value_lower_bound": True
}
