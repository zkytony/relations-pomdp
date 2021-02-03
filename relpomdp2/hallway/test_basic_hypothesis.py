"""This script tests the hypothesis that
an agent with the ability to perceive correlated
object should have a policy that obtains better
reward because of its ability to use that perception
ability.

This is for the multi-object (>= 3) case. The
number of objects can be specified as a variable
"""

from relpomdp2.hallway.problem import *
import random
import copy
import numpy as np
import pandas as pd
import time
import os
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

def make_actions(detectable_object_names):
    actions = [HSAction("left"), HSAction("right"),
               HSAction("Declare")]
    for obj in detectable_object_names:
        actions.append(HSAction("look-%s" % obj))
    return actions

def chmap(values):
    return {id2ch(i):values[i]
            for i in range(len(values))}

def fpfn(fp, fn):
    return {"FP": fp, "FN": fn}

def sensor_cfg_to_str(sensor_ranges,
                      sensor_costs,
                      sensor_noises):
    res = []
    for obj in sensor_ranges:
        res.append("%sr%dc%dn%.2f%.2f" % (obj,
                                          sensor_ranges[obj],
                                          sensor_costs[obj],
                                          sensor_noises[obj]["FP"],
                                          sensor_noises[obj]["FN"]))
    return "-".join(res)

# Params
def RUN(HALLWAY_LEN = 6,
        NOBJS = 2,
        SPATIAL_CORR_FUNC = spatially_close,
        SENSOR_RANGES = chmap([0, 1]),
        SENSOR_COSTS = chmap([0, 0]),
        SENSOR_NOISES = chmap([fpfn(0.0, 0.0),
                               fpfn(0.0, 0.0)]),
                               # fpfn(0.0, 0.0)])
        PAIRWISE_RELATIONS = {("A", "B"): spatially_close},
        ACTIONS_ONLY_TARGET = make_actions(["A"]),
        ACTIONS_BOTH_DETECTABLE = make_actions(["A", "B"]),
        NUM_STEPS = 100,
        NTRIALS = 1000,
        DISCOUNT_FACTOR=0.95,
        SARSOP_MEMORY = 200, # MB
        SARSOP_TIME = 180, # seconds
        SARSOP_PRECISION=1e-12,
        USE_CORRELATION_PRIOR=True,
        DEBUGGING=False):
    """Run one round: based on the configuraiton,
    first compute the SARSOP policy for the target-only
    detector agent, and the both target detector agent.

    For either agent, they are aware that the environment
    contains not just the target. So you can specify them
    to use the joint distirbution as prior (set USE_CORRELATION_PRIOR to be True)"""
    suffix = "%d_%d_%s_%s_uc%s" %\
             (HALLWAY_LEN, NOBJS, SPATIAL_CORR_FUNC.__name__,
              sensor_cfg_to_str(SENSOR_RANGES, SENSOR_COSTS, SENSOR_NOISES),
              str(USE_CORRELATION_PRIOR))

    os.makedirs("results", exist_ok=True)

    policies_filename = "policies_%s.pkl" % suffix
    policies_both_detectable = {}
    policies_only_target = {}
    init_belief_both_detectable = {}
    init_belief_only_target = {}
    if os.path.exists(os.path.join("results", policies_filename)):
        with open(os.path.join("results", policies_filename), "rb") as f:
            print("LOADING %s" % policies_filename)
            policies_only_target, policies_both_detectable,\
                init_belief_only_target, init_belief_both_detectable = pickle.load(f)
    else:
        # Obtain the policy for each initial robot position
        for r in range(HALLWAY_LEN):
            print("********** COMPUTING POLICY FOR robot location = %d **********" % r)
            init_state = random_init_state(NOBJS, HALLWAY_LEN, init_r=r)

            instance_config = {"pairwise_relations": PAIRWISE_RELATIONS,
                               "sensor_ranges": SENSOR_RANGES,
                               "sensor_costs": SENSOR_COSTS,
                               "sensor_noises": SENSOR_NOISES}
            solver_config = {
                "timeout": SARSOP_TIME,
                "discount_factor": DISCOUNT_FACTOR,
                "memory": SARSOP_MEMORY,
                "precision": SARSOP_PRECISION,
            }

            # only target detectable (no correlation considered)
            agent, env = create_instance((init_state, HALLWAY_LEN),
                                         use_correlation_prior=USE_CORRELATION_PRIOR,
                                         actions=ACTIONS_ONLY_TARGET,
                                         **instance_config)
            policies_only_target[r], _ = compute_policy(
                agent, env,
                logfile=os.path.join("results", "policy_only_target_%s.log" % suffix),
                **solver_config)
            init_belief_only_target[r] = copy.deepcopy(agent.belief)
            # both detectable
            agent, env = create_instance((init_state, HALLWAY_LEN),
                                         use_correlation_prior=USE_CORRELATION_PRIOR,
                                         actions=ACTIONS_BOTH_DETECTABLE,
                                         **instance_config)
            policies_both_detectable[r], _ = compute_policy(
                agent, env,
                logfile=os.path.join("results", "policy_both_detectable_%s.log" % suffix),
                **solver_config)
            init_belief_both_detectable[r] = copy.deepcopy(agent.belief)

    with open(os.path.join("results", policies_filename), "wb") as f:
        pickle.dump((policies_only_target, policies_both_detectable,
                     init_belief_only_target, init_belief_both_detectable), f)

    data = []
    _init_a_counts = [0] * HALLWAY_LEN
    for t in range(NTRIALS):
        print("********** TRIAL %d/%d *********" % (t+1, NTRIALS))
        locations = set(i for i in range(HALLWAY_LEN))
        init_r = random.sample(locations, 1)[0]

        # Note that this initial state DOES use pairwise relations!
        # But the agent may or may not begin with a belief distribution
        # conditioned on such relation.
        init_state = random_init_state(NOBJS, HALLWAY_LEN,
                                       pairwise_relations=PAIRWISE_RELATIONS,
                                       init_r=init_r)

        instance_config = {"pairwise_relations": PAIRWISE_RELATIONS,
                           "sensor_ranges": SENSOR_RANGES,
                           "sensor_costs": SENSOR_COSTS,
                           "sensor_noises": SENSOR_NOISES}
        sim_config = {"viz": False,
                      "num_steps": NUM_STEPS,
                      "discount_factor": 0.95}

        # ONLY TARGET
        agent, env = create_instance((init_state, HALLWAY_LEN),
                                     use_correlation_prior=USE_CORRELATION_PRIOR,
                                     actions=ACTIONS_ONLY_TARGET,
                                     **instance_config)

        policy = policies_only_target[init_r]
        value = policy.value(agent.belief)
        cum_reward, _history, _meta = simulate_policy(policy, agent, env,
                                                      **sim_config)
        data.append([t, str(init_state), "A", cum_reward, value, _meta["odiff_count"]])

        ### for debugging the instance distribution
        if DEBUGGING:
            init_belief = init_belief_only_target[init_r]
            a_probs = [init_belief[s]
                       for s in agent.all_states
                       if s.r == init_state.r]
            _init_a_counts[init_state.x] += 1
            _init_a_dist = [_init_a_counts[i] / sum(_init_a_counts)
                            for i in range(len(_init_a_counts))]
            # This is printed for every trial. Only the last one counts.
            print("Actual distribution:", _init_a_dist)
            print("Distribution used for SARSOP:", a_probs)
            print("(they should differ. SARSOP should be uniform."
                  "Actual distribution should not (because it is using correlation)")

        # BOTH OBJECTS
        policy = policies_both_detectable[init_r]
        agent, env = create_instance((init_state, HALLWAY_LEN),
                                     use_correlation_prior=USE_CORRELATION_PRIOR,
                                     actions=ACTIONS_BOTH_DETECTABLE,
                                     **instance_config)
        value = policy.value(agent.belief)
        cum_reward, _history, _meta = simulate_policy(policy, agent, env,
                                                      **sim_config)
        data.append([t, str(init_state), "AB", cum_reward, value, _meta["odiff_count"]])

        ### for debugging the instance distribution
        if DEBUGGING:
            init_belief = init_belief_both_detectable[init_r]
            a_probs = [init_belief[s]
                       for s in agent.all_states
                       if s.r == init_state.r]
            _init_a_counts[init_state.x] += 1
            _init_a_dist = [_init_a_counts[i] / sum(_init_a_counts)
                            for i in range(len(_init_a_counts))]
            # This is printed for every trial. Only the last one counts.
            print("Actual distribution:", _init_a_dist)
            print("Distribution used for SARSOP:", a_probs)
            print("(they should be very close. Both are using correlation)")

        # BOTH Objects, but force lookB first.
        policy = policies_both_detectable[init_r]
        agent, env = create_instance((init_state, HALLWAY_LEN),
                                     use_correlation_prior=USE_CORRELATION_PRIOR,
                                     actions=ACTIONS_BOTH_DETECTABLE,
                                     **instance_config)
        value = policy.value(agent.belief)
        cum_reward, _history, _meta = simulate_policy(policy, agent, env,
                                                      hardcode_plan=[HSAction("look-B")],
                                                      **sim_config)
        data.append([t, str(init_state), "AB-manual", cum_reward, value, _meta["odiff_count"]])


    df = pd.DataFrame(data, columns=["trial_no", "setting", "Detectable", "return", "Vb0", "odiff_count"])
    timestamp = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
    df.to_csv(os.path.join("results", "results_%s_%s.csv" % (suffix, timestamp)))

    print(df.groupby("Detectable").agg(["mean","std"]))
    return df

if __name__ == "__main__":
    RUN(HALLWAY_LEN = 4,
        NOBJS = 2,
        SPATIAL_CORR_FUNC = spatially_close,
        SENSOR_RANGES = chmap([0, 1]),
        SENSOR_COSTS = chmap([0, 0]),
        SENSOR_NOISES = chmap([fpfn(0.0, 0.0),
                               fpfn(0.0, 0.0)]),
                               # fpfn(0.0, 0.0)])
        PAIRWISE_RELATIONS = {("A", "B"): spatially_close},
        ACTIONS_ONLY_TARGET = make_actions(["A"]),
        ACTIONS_BOTH_DETECTABLE = make_actions(["A", "B"]),
        NUM_STEPS = 100,
        NTRIALS = 500,
        DISCOUNT_FACTOR=0.95,
        SARSOP_MEMORY = 200, # MB
        SARSOP_TIME = 180, # seconds
        SARSOP_PRECISION=1e-12,
        USE_CORRELATION_PRIOR=True,
        DEBUGGING=True)
