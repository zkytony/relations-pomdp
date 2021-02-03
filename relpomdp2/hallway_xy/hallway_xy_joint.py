# Same domain as hallway_xy except the
# agent now tracks both the states of X and Y

import os
import pomdp_py
import time
import math
import copy
import random
from relpomdp2.probability import TabularDistribution
from relpomdp2.constants import SARSOP_PATH, VI_PATH
import matplotlib.pyplot as plt
import seaborn as sns
from hallway_xy import HSTrueState, HSAction, HSObservation,\
    HSTransitionModel, HSTrueObservationModel, HSRewardModel, HSAgent, HSEnv,\
    ACTIONS, OBSERVATIONS, spatially_apart, spatially_close, spatially_exact,\
    spatially_independent, str_to_setting, indicator

class HSTrueTransitionModel(pomdp_py.DetTransitionModel):
    def __init__(self, hallway_length):
        self.hallway_length = hallway_length
        super().__init__(epsilon=0.0)

    def sample(self, state, action):
        if action.name == "left":
            r = state.r - 1
        elif action.name == "right":
            r = state.r + 1
        else:
            r = state.r
        r = max(0, min(self.hallway_length-1, r))
        return HSTrueState(r, state.x, state.y)

    def get_all_states(self):
        return [HSTrueState(r, x, y)
                for r in range(self.hallway_length)
                for x in range(self.hallway_length)
                for y in range(self.hallway_length)]


class HSTrueAgent(pomdp_py.Agent):
    def __init__(self,
                 hallway_length,
                 init_belief,
                 range_x,
                 range_y,
                 actions=ACTIONS):
        policy_model = pomdp_py.UniformPolicyModel(actions)
        transition_model = HSTrueTransitionModel(hallway_length)
        observation_model = HSTrueObservationModel(range_x, range_y)
        reward_model = HSRewardModel()
        self.hallway_length = hallway_length
        super().__init__(init_belief,
                         policy_model,
                         transition_model,
                         observation_model,
                         reward_model)


def main(setting="YR..X",
         spatial_corr_func=spatially_close,
         range_x=0, range_y=1, actions=ACTIONS,
         pomdp_name="hallway_search_joint",
         reuse_policy=False, use_correlation=True):
    true_r, true_x, true_y, hallway_length = str_to_setting(setting)
    init_state = HSTrueState(true_r, true_x, true_y)
    env = HSEnv(init_state, hallway_length)

    # correlation joint probability
    variables = ["X", "Y"]
    weights = [
        ((x, y), indicator(spatial_corr_func(x, y)))
        for x in range(hallway_length)
        for y in range(hallway_length)
    ]
    joint_dist = TabularDistribution(variables, weights)

    # Create belief
    agent = HSTrueAgent(hallway_length,
                        None,
                        range_x,
                        range_y,
                        actions=actions)
    # uniform belief over x and y but true belief over r.
    init_belief_hist = {}
    total_prob = 0.0
    for state in agent.transition_model.get_all_states():
        if state.r == true_r:
            if use_correlation:
                init_belief_hist[state] = joint_dist.prob((state.x, state.y))
            else:
                init_belief_hist[state] = 1.0
        else:
            init_belief_hist[state] = 0.0
        total_prob += init_belief_hist[state]
    # Normalize
    for state in init_belief_hist:
        init_belief_hist[state] /= total_prob
    agent.set_belief(pomdp_py.Histogram(init_belief_hist), prior=True)

    pomdp_filename = "%s.%s.%s.%s" % (pomdp_name,
                                      setting.replace("X", ".").replace("Y", ".").replace(".", "x"),
                                      spatial_corr_func.__name__, "using" if use_correlation else "not")

    discount_factor = 0.95
    _num_steps = 10
    if reuse_policy and os.path.exists("./%s.policy" % pomdp_filename):
        policy = pomdp_py.AlphaVectorPolicy.construct("./%s.policy" % pomdp_filename,
                                                      agent.all_states, agent.all_actions,
                                                      solver="sarsop")
    else:
        policy = pomdp_py.sarsop(agent, SARSOP_PATH,
                                 discount_factor=discount_factor, timeout=20,
                                 memory=20, precision=1e-12,
                                 pomdp_name=pomdp_filename,
                                 remove_generated_files=False)
    # # technically a planner
    # policy = pomdp_py.POUCT(max_depth=10, num_sims=2000,
    #                         discount_factor=discount_factor,
    #                         exploration_const=100,
    #                         rollout_policy=agent.policy_model)
    return simulate_policy(policy, agent, env,
                           num_steps=_num_steps, discount_factor=discount_factor)

def simulate_policy(policy, agent, env, num_steps=10, discount_factor=0.95):
    _cum_reward = 0.0
    _discount = 1.0
    init_belief_hist = copy.deepcopy(agent.belief.get_histogram())
    for step in range(num_steps):
        action = policy.plan(agent)
        reward = env.state_transition(action, execute=True)
        observation = agent.observation_model.sample(env.state, action)
        env.print_state()

        # policy and belief update
        policy.update(agent, action, observation)  # only useful for pouct
        new_belief = pomdp_py.update_histogram_belief(agent.cur_belief,
                                                      action, observation,
                                                      agent.observation_model,
                                                      agent.transition_model)
        agent.set_belief(new_belief)

        # Simulating reward
        _cum_reward += _discount * reward
        _discount *= discount_factor
        print("[step=%d] action=%s, observation=%s, reward=%d, cum_reward=%.5f"\
              % (step+1, action, observation, reward, _cum_reward))

        # Termination
        if action.name == "Declare":
            print("Done. Discounted cumulative reward: %.5f" % _cum_reward)
            break

    plt.clf()
    # Return many things for further analysis
    return (policy, agent, pomdp_py.Histogram(init_belief_hist), _cum_reward)

def with_and_without(setting, spatial_corr, range_x=0, range_y=1):
    """Run agent in `setting` with SARSOP policy computed
    for the case with lookY in the action space and for
    the case without lookY in the action space."""
    policy_NOlookY, _, init_belief1, cum_rewad1 = main(
        setting,
        spatial_corr_func=eval(spatial_corr),
        range_x=range_x, range_y=range_y,
        pomdp_name="_hallway_search_joint",
        reuse_policy=True,
        use_correlation=False,
        actions=[HSAction("left"), HSAction("right"), HSAction("lookX"),
                 HSAction("Declare")])

    policy_HASlookY, _, init_belief2, cum_rewad2 = main(
        setting,
        spatial_corr_func=eval(spatial_corr),
        range_x=range_x, range_y=range_y,
        pomdp_name="_hallway_search_joint",
        reuse_policy=True,
        use_correlation=True,
        actions=[HSAction("left"), HSAction("right"), HSAction("lookX"),
                 HSAction("lookY"), HSAction("Declare")])

    # Note that when correlation is considered (WITH lookY), the
    # initial belief is not the same as when not. We could, you know,
    # consider the whole joint distribution as an observation that
    # occurs at the first step. Therefore we evaluate at init_belief1
    print("Value at initial belief WITH lookY: {}".format(policy_HASlookY.value(init_belief2)))
    print("Value at initial belief WITHOUT lookY: {}".format(policy_NOlookY.value(init_belief1)))


if __name__ == "__main__":
    # policy, agent, init_belief = main(setting="XY..R",
    #                                   spatial_corr_func=spatially_close,
    #                                   range_x=0, range_y=1, actions=ACTIONS,
    #                                   pomdp_name="hallway_search_joint")

    print("------Spatially E X A C T-------------")
    with_and_without("XY.R", "spatially_exact")
    print("---------Spatially A P A R T------------")
    print(" =========== 1 =========")
    with_and_without("X.YR", "spatially_apart")
    print(" =========== 2 =========")
    with_and_without("XR.Y", "spatially_apart")
    print("---------Spatially C L O S E------------")
    print(" =========== 1 =========")
    with_and_without("XY.R", "spatially_close")
    print(" =========== 2 =========")
    with_and_without("XY.R.", "spatially_close")
    print("---------Spatially I N D E P------------")
    with_and_without("XY.R", "spatially_independent")
