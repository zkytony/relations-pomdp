"""
The value of SARSOP policy at initial belief
should be equal to the actual return, if:
* There is no observation noise
* The distribution of instances used to get actual return
  is exactly the same as the initial belief
We use the hallway domain. Assume there is no noise
in the target detector. And we manually make sure
the second condition is true. Then we verify if
the value equals to the return
"""
from relpomdp2.constants import SARSOP_PATH, VI_PATH
from relpomdp2.hallway.problem import create_instance, simulate_policy,\
    HSJointState, HSEnv, HSObservationModel
from relpomdp2.hallway.test_basic_hypothesis import make_actions
import pomdp_py
import random
import numpy as np
import copy

# Parameters
PARAMS = dict(
    sensor_ranges={"A":0},
    sensor_costs={"A":0},
    sensor_noises={"A":{"FP":0.0, "FN":0.0}},
    pairwise_relations={}
)
SOLVER_PARAMS = dict(
    timeout=20,
    memory=20,
    precision=1e-12,
    logfile=None
)
NTRIALS = 500

# Compute a policy
setting = "R..A"
agent, env = create_instance(setting,
                             actions=make_actions(["A"]),
                             **PARAMS)
init_belief = copy.deepcopy(agent.belief)
discount_factor = 0.95
policy = pomdp_py.sarsop(
    agent, SARSOP_PATH,
    discount_factor=discount_factor,
    **SOLVER_PARAMS)

# The distribution of object A that the agent believes
a_probs = [init_belief[s]
           for s in agent.all_states
           if s.r == env.state.r]
# If you set a_probs[0] = 0, you will get the same behavior as in the experiments,
# where value =~ 8.29 but the actual_return =~7.9. With the default setting, however,
# the value =~ 8.29, with some standard deviation.
print(a_probs)

# We want our trials to have the same starting robot location
init_r = env.state.r
_rewards = []
_init_a_counts = [0] * len(a_probs)  # tracks the actual distribution
for t in range(NTRIALS):
    print("~~~~~~~~~~~~ TRIAL %d/%d ~~~~~~~~~~~~~~" % (t+1, NTRIALS))
    # We construct an environment that follows the a_probs distribution
    agent.set_belief(init_belief)
    init_a = random.choices(np.arange(len(setting)), weights=a_probs, k=1)[0]
    init_state = HSJointState(init_r, (init_a,))
    env = HSEnv(init_state, len(setting), PARAMS["sensor_costs"])
    env.true_observation_model = HSObservationModel(PARAMS["sensor_ranges"],
                                                    PARAMS["sensor_noises"])
    _cum_reward, _, _ = simulate_policy(policy, agent, env, viz=False,
                                        num_steps=100, discount_factor=discount_factor)
    _rewards.append(_cum_reward)
    _init_a_counts[init_a] += 1

# Report
print(np.mean(_rewards), np.std(_rewards))
print(policy.value(init_belief))
_init_a_dist = [_init_a_counts[i] / sum(_init_a_counts)
                for i in range(len(_init_a_counts))]
print("Expected distribution: {}".format(a_probs))
print("Actual distribution: {}".format(_init_a_dist))
