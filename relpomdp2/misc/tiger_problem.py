"""The classic Tiger problem.

This is a POMDP problem; Namely, it specifies both
the POMDP (i.e. state, action, observation space)
and the T/O/R for the agent as well as the environment.

The description of the tiger problem is as follows: (Quote from `POMDP:
Introduction to Partially Observable Markov Decision Processes
<https://cran.r-project.org/web/packages/pomdp/vignettes/POMDP.pdf>`_ by
Kamalzadeh and Hahsler )

A tiger is put with equal probability behind one
of two doors, while treasure is put behind the other one.
You are standing in front of the two closed doors and
need to decide which one to open. If you open the door
with the tiger, you will get hurt (negative reward).
But if you open the door with treasure, you receive
a positive reward. Instead of opening a door right away,
you also have the option to wait and listen for tiger noises. But
listening is neither free nor entirely accurate. You might hear the
tiger behind the left door while it is actually behind the right
door and vice versa.

States: tiger-left, tiger-right
Actions: open-left, open-right, listen
Rewards:
    +10 for opening treasure door. -100 for opening tiger door.
    -1 for listening.
Observations: You can hear either "tiger-left", or "tiger-right".

Note that in this example, the TigerProblem is a POMDP that
also contains the agent and the environment as its fields. In
general this doesn't need to be the case. (Refer to more complicated
examples.)

"""

import pomdp_py
import random
import numpy as np
import sys
from relpomdp2.constants import SARSOP_PATH, VI_PATH

def indicator(cond):
    if cond:
        return 1.0
    else:
        return 0.0

class TigerState(pomdp_py.SimpleState):
    def __init__(self, name):
        self.name = name
        super().__init__(name)

class TigerAction(pomdp_py.SimpleAction):
    def __init__(self, name):
        super().__init__(name)

class TigerObservation(pomdp_py.SimpleObservation):
    def __init__(self, name):
        self.name = name
        super().__init__(name)

class TigerTransitionModel(pomdp_py.TransitionModel):
    STATES = {TigerState("tiger-left"),
              TigerState("tiger-middle"),
              TigerState("tiger-right")}

    def probability(self, next_state, state, action):
        if action.name.startswith("open"):
            return 1.0 / len(TigerTransitionModel.STATES)
        else:
            return indicator(next_state.name == state.name)

    def get_all_states(self):
        """Only need to implement this if you're using
        a solver that needs to enumerate over the observation space (e.g. value iteration)"""
        return TigerTransitionModel.STATES

class TigerObservationModel(pomdp_py.ObservationModel):
    OBSERVATIONS = {TigerObservation("tiger-left"),
                    TigerObservation("tiger-middle"),
                    TigerObservation("tiger-right")}

    def __init__(self, noise=0.15):
        self.noise = noise

    def probability(self, observation, next_state, action):
        if action.name == "listen":
            if observation.name == next_state.name: # heard the correct growl
                return 1.0 - self.noise
            else:
                # Equally likely for the remaining states
                return self.noise / (len(TigerObservationModel.OBSERVATIONS) - 1)
        else:
            return 1.0 / len(TigerObservationModel.OBSERVATIONS)

    def sample(self, next_state, action):
        probs = []
        observations = list(TigerObservationModel.OBSERVATIONS)
        for o in observations:
            probs.append(self.probability(o, next_state, action))
        return random.choices(observations, weights=probs, k=1)[0]

    def get_all_observations(self):
        """Only need to implement this if you're using
        a solver that needs to enumerate over the observation space (e.g. value iteration)"""
        return TigerObservationModel.OBSERVATIONS

# Reward Model
class TigerRewardModel(pomdp_py.RewardModel):
    def _reward_func(self, state, action):
        if action.name.startswith("open"):
            side = action.name.split("-")[1]
            if not state.name.endswith(side):
                return -20
            else:
                return 10
        else:
            return 0

    def sample(self, state, action, next_state):
        # deterministic
        return self._reward_func(state, action)

# Policy Model
class TigerPolicyModel(pomdp_py.RandomRollout):
    """This is an extremely dumb policy model; To keep consistent
    with the framework."""
    # A stay action can be added to test that POMDP solver is
    # able to differentiate information gathering actions.
    ACTIONS = {TigerAction(s) for s in {"open-left",
                                        "open-right",
                                        "open-middle",
                                        "listen"}}
    def sample(self, state, **kwargs):
        return self.get_all_actions().random()

    def get_all_actions(self, **kwargs):
        return TigerPolicyModel.ACTIONS


class TigerProblem(pomdp_py.POMDP):
    """
    In fact, creating a TigerProblem class is entirely optional
    to simulate and solve POMDPs. But this is just an example
    of how such a class can be created.
    """

    def __init__(self, obs_noise, init_true_state, init_belief):
        """init_belief is a Distribution."""
        agent = pomdp_py.Agent(init_belief,
                               TigerPolicyModel(),
                               TigerTransitionModel(),
                               TigerObservationModel(obs_noise),
                               TigerRewardModel())
        env = pomdp_py.Environment(init_true_state,
                                   TigerTransitionModel(),
                                   TigerRewardModel())
        super().__init__(agent, env, name="TigerProblem")

    @classmethod
    def create_instance(cls, noise=0.15,
                        init_state="tiger-left",
                        init_belief="uniform"):
        if init_belief == "uniform":
            b0 = {}
            for state in TigerTransitionModel.STATES:
                b0[state] = 1.0 / len(TigerTransitionModel.STATES)
            init_belief = pomdp_py.Histogram(b0)
        if type(init_state) == str:
            init_state = TigerState(init_state)
        return TigerProblem(noise, init_state, init_belief)


def test_planner(tiger_problem, planner, nsteps=3):
    """
    Runs the action-feedback loop of Tiger problem POMDP

    Args:
        tiger_problem (TigerProblem): an instance of the tiger problem.
        planner (Planner): a planner
        nsteps (int): Maximum number of steps to run this loop.
    """
    for i in range(nsteps):
        action = planner.plan(tiger_problem.agent)
        print("==== Step %d ====" % (i+1))
        print("True state: %s" % tiger_problem.env.state)
        print("Belief: %s" % str(tiger_problem.agent.cur_belief))
        print("Action: %s" % str(action))
        print("Reward: %s" % str(tiger_problem.env.reward_model.sample(tiger_problem.env.state, action, None)))

        # Let's create some simulated real observation; Update the belief
        # Creating true observation for sanity checking solver behavior.
        # In general, this observation should be sampled from agent's observation model.
        real_observation = tiger_problem.env.provide_observation(
            tiger_problem.agent.observation_model, action)
        print(">> Observation: %s" % real_observation)
        tiger_problem.agent.update_history(action, real_observation)

        planner.update(tiger_problem.agent, action, real_observation)
        if isinstance(planner, pomdp_py.POUCT):
            print("Num sims: %d" % planner.last_num_sims)
            print("Plan time: %.5f" % planner.last_planning_time)
        if isinstance(tiger_problem.agent.cur_belief, pomdp_py.Histogram):
            new_belief = pomdp_py.update_histogram_belief(tiger_problem.agent.cur_belief,
                                                          action, real_observation,
                                                          tiger_problem.agent.observation_model,
                                                          tiger_problem.agent.transition_model)
            tiger_problem.agent.set_belief(new_belief)
        if action.name.startswith("open"):
            # Make it clearer to see what actions are taken until every time door is opened.
            print("\n")

def main():
    TigerTransitionModel.STATES = {TigerState("tiger-left"),
                                   TigerState("tiger-hak"),
                                   TigerState("tiger-right")}
    tiger_problem = TigerProblem.create_instance(0.15)

    print("** Testing value iteration **")
    vi = pomdp_py.vi_pruning(tiger_problem.agent, VI_PATH,
                             discount_factor=0.95,
                             options=["-horizon", 10],
                             # memory=20, precision=1e-12, timeout=20,
                             pomdp_name="tiger",
                             remove_generated_files=True)
    test_planner(tiger_problem, vi, nsteps=3)

if __name__ == '__main__':
    main()
