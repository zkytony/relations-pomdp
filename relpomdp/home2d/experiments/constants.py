

METHOD_TO_NAME = {
    "mdp" : "MDP(full)",
    "pomdp": "NS(full)",
    "pomdp-subgoal": "S+B(full)",
    "pomdp-nk": "NS",  # no subgoal
    "pomdp-subgoal-nk": "S+B",   # subgoal + belief update
    "pomdp-subgoal-nk-nocorr": "S",   # subgoal only
    "random-nk": "Rand",    # random
    "heuristic-nk": "Heur"  # heuristic
}
