"""Object search in a hallway. We call this domain HallwaySearch

X . . robot . . Y

X and Y are the locations of two objects.
They follow a joint distribution:

   Their locations are never within a diameter of 2 of each other.

The robot has five actions:

   LookX: Fires a detector for X. Only able to look down at where the robot is
   LookY: Fires a detector for Y. Able to check if Y is present in the grid
          cells next to the robot
   Left: Move left 1 cell
   Right: Move right 1 cell
   Declare: Declares the location of X to be found

The observations are:

   Nothing: did not detect any object
   SeeX: Detected X
   SeeY: Detected Y

The transitions are deterministic.

The rewards are:

   +10: Declare successful
   -10: Decalre failed
     0: Otherwise

We experiment with two kinds of state space. First, both X and Y are modeled.
Second, only X is modeled. In the implementation below, we try the second one first.

We use SARSOP or Value Iteration (by Anthony Cassandra) to solve these problems
They should be solvable since it is quite small.
"""

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


###### State, Action, Observation #######
class HSTrueState(pomdp_py.SimpleState):
    def __init__(self, r, x, y):
        """
        True state. Used by environment
        Args:
            r (int): Location of robot
            x (int): Location of target object
        """
        self.r = r
        self.x = x
        self.y = y
        super().__init__((r,x,y))

    def __str__(self):
        return "r%dx%dy%d" % (self.r, self.x, self.y)

class HSState(pomdp_py.SimpleState):
    def __init__(self, r, x):
        """
        Agent state, used in agent belief
        Args:
            r (int): Location of robot
            x (int): Location of target object
        """
        self.r = r
        self.x = x
        super().__init__((r,x))

    def __str__(self):
        return "r%dx%d" % (self.r, self.x)

class HSAction(pomdp_py.SimpleAction):
    def __init__(self, name):
        super().__init__(name)

class HSObservation(pomdp_py.SimpleObservation):
    def __init__(self, value):
        self.value = value
        super().__init__(value)

# Action space, observation space
ACTIONS = [HSAction("left"), HSAction("right"), HSAction("lookY"),
           HSAction("lookX"), HSAction("Declare")]
OBSERVATIONS = [HSObservation("nothing"), HSObservation("X"), HSObservation("Y")]

######### Transition, Observation, Reward models #######
class HSTransitionModel(pomdp_py.DetTransitionModel):
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
        if isinstance(state, HSState):
            return HSState(r, state.x)
        elif isinstance(state, HSTrueState):
            return HSTrueState(r, state.x, state.y)

    def get_all_states(self):
        return [HSState(r, x)
                for r in range(self.hallway_length)
                for x in range(self.hallway_length)]

def in_range(r, x, range_size):
    return abs(r - x) <= range_size

def indicator(cond):
    if cond:
        return 1.0
    else:
        return 0.0

class HSObservationModel(pomdp_py.ObservationModel):
    def __init__(self, joint_dist, range_x, range_y):
        """
        Args:
            joint_dist (TabularDistribution): Specifies the joint
                distribution of object X and Y
            rangex (int): The range of detection for X
            rangey (int): The range of detection for Y
        """
        self.joint_dist = joint_dist
        self.range_x = range_x
        self.range_y = range_y
        self._cache = {}  # cache for sample

    def _sensor_model(self, r, obj_loc, obj_name, oval):
        if obj_name == "X":
            if oval == "X":
                return indicator(in_range(r, obj_loc, self.range_x))
            elif oval == "nothing":
                return indicator(not in_range(r, obj_loc, self.range_x))
            else: # observation.value == "Y":
                return 0.0
        elif obj_name == "Y":
            if oval == "Y":
                return indicator(in_range(r, obj_loc, self.range_y))
            elif oval == "nothing":
                return indicator(not in_range(r, obj_loc, self.range_y))
            else: # observation.value == "X":
                return 0.0

    def _corr_model(self, r, x, oval):
        """
        Returns the probility of observing oval (either nothing, or Y)
        given the location of the robot and the location of x
        """
        assert oval in {"nothing", "Y"}
        cond_dist = self.joint_dist.condition({"X": x})
        prob = 0.0
        no_probable_y = True
        for y in self.joint_dist.ranges["Y"]:
            prob += self._sensor_model(r, y, "Y", oval) * cond_dist.prob({"Y":y})
            if cond_dist.prob({"Y":y}) > 0.0:
                no_probable_y = False
        if no_probable_y:
            # There is no where for y to be so we will definitely see nothing
            return indicator(oval == "nothing")
        return prob

    def probability(self, observation, next_state, action):
        r, x = next_state.r, next_state.x
        if action.name == "lookX":
            return self._sensor_model(r, x, "X", observation.value)

        elif action.name == "lookY":
            if observation.value == "X":
                return 0.0
            else:
                return self._corr_model(r, x, observation.value)

        else:
            return indicator(observation.value == "nothing")

    def sample(self, next_state, action):
        if (next_state, action) in self._cache:
            probs = self._cache[(next_state, action)]
        else:
            probs = []
            for o in OBSERVATIONS:
                probs.append(self.probability(o, next_state, action))
        return random.choices(OBSERVATIONS, weights=probs, k=1)[0]

    def get_all_observations(self):
        return OBSERVATIONS


class HSTrueObservationModel(HSObservationModel):
    """This is the observation model that
    is conditioned on the true state. Only sensor model is used
    to calculate probability and sampling"""
    def __init__(self, range_x, range_y):
        self.range_x = range_x
        self.range_y = range_y

    def probability(self, observation, next_state, action):
        # Only needs to use sensor model. No correlation needed (already
        # incorporated into the initial belief.)
        r, x, y = next_state.r, next_state.x, next_state.y
        if action.name == "lookX":
            return self._sensor_model(r, x, "X", observation.value)
        elif action.name == "lookY":
            return self._sensor_model(r, y, "Y", observation.value)
        else:
            return indicator(observation.value == "nothing")


    def sample(self, next_state, action):
        """Deterministic sensor model"""
        r, x, y = next_state.r, next_state.x, next_state.y
        if action.name == "lookX":
            for oval in {"X", "nothing"}:
                if self._sensor_model(r, x, "X", oval) > 0:
                    return HSObservation(oval)
            assert False, "This should not happen"
        elif action.name == "lookY":
            for oval in {"Y", "nothing"}:
                if self._sensor_model(r, y, "Y", oval) > 0:
                    return HSObservation(oval)
            assert False, "This should not happen"
        else:
            return HSObservation("nothing")


class HSRewardModel(pomdp_py.DetRewardModel):
    def __init__(self, cost_x=0, cost_y=0):
        self.cost_x = cost_x
        self.cost_y = cost_y

    def reward_func(self, state, action, next_state):
        if action.name == "Declare":
            if state.r == state.x:
                return 10
            else:
                return -10
        else:
            if action.name == "lookX":
                return self.cost_x
            elif action.name == "lookY":
                return self.cost_y
            return 0


class HSAgent(pomdp_py.Agent):
    def __init__(self,
                 hallway_length,
                 init_belief,
                 joint_dist,
                 range_x,
                 range_y,
                 actions=ACTIONS):
        policy_model = pomdp_py.UniformPolicyModel(actions)
        transition_model = HSTransitionModel(hallway_length)
        observation_model = HSObservationModel(joint_dist, range_x, range_y)
        reward_model = HSRewardModel()
        self.hallway_length = hallway_length
        super().__init__(init_belief,
                         policy_model,
                         transition_model,
                         observation_model,
                         reward_model)

class HSEnv(pomdp_py.Environment):
    def __init__(self,
                 init_state, hallway_length):
        transition_model = HSTransitionModel(hallway_length)
        reward_model = HSRewardModel()
        self.hallway_length = hallway_length
        super().__init__(init_state, transition_model, reward_model)

    def print_state(self):
        output = ""
        for i in range(self.hallway_length):
            if self.state.r == i:
                output += "R"
            elif self.state.x == i:
                output += "X"
            elif  self.state.y == i:
                output += "Y"
            else:
                output += "."
        print(output)


def spatially_correlated(x, y):
    return abs(x - y) > 1 #abs(x - y) <= 2

def spatially_apart(x, y):
    return abs(x - y) > 1

def spatially_close(x, y):
    return abs(x - y) <= 1

def spatially_exact(x, y):
    return abs(x - y) == 2

def spatially_independent(x, y):
    return True

def plot_belief(belief, ax, hallway_length):
    bx = []
    xvals = []
    for state in belief:
        if belief[state] > 0.0:
            xvals.append(state.x)
            bx.append(belief[state])
    ax.bar(xvals, bx, label="$b(X)$")
    ax.set_ylim(0, 1)
    ax.set_xlim(-1, hallway_length)
    ax.set_xticks([i for i in range(hallway_length)])

def plot_true_state(true_state, ax):
    r, x, y = true_state.r, true_state.x, true_state.y
    ax.plot([r], 0.05, "ro", markersize=10, label="robot")
    ax.plot([x], 0.05, marker="$X$", color='black', markersize=10, label="X")
    ax.plot([y], 0.05, marker="$Y$", color='black', markersize=10, label="Y")
    ax.legend(loc='upper left')

def str_to_setting(situation):
    true_r = situation.index("R")
    true_x = situation.index("X")
    true_y = situation.index("Y")
    hallway_length = len(situation)
    return true_r, true_x, true_y, hallway_length

def random_setting(hallway_length,
                   init_r=None,
                   spatial_corr_func=None):
    """Returns a string representation of a random setting"""
    locations = set(i for i in range(hallway_length))
    if init_r is None:
        init_r = random.sample(locations, 1)[0]
    setting = ["."] * hallway_length
    setting[init_r] = "R"
    count = 0
    while count < 1000:
        x, y = random.sample(locations - {init_r}, 2)
        if spatial_corr_func is not None:
            if spatial_corr_func(x,y):
                break
        else:
            break
        count += 1
    setting[x] = "X"
    setting[y] = "Y"
    return "".join(setting)

def initialize_setting(setting,
                       range_x=0, range_y=1,
                       spatial_corr_func=spatially_correlated,
                       actions=ACTIONS):
    true_r, true_x, true_y, hallway_length = str_to_setting(setting)

    init_state = HSTrueState(true_r, true_x, true_y)
    env = HSEnv(init_state, hallway_length)
    true_observation_model = HSTrueObservationModel(range_x, range_y)
    env.true_observation_model = true_observation_model

    # correlation joint probability
    variables = ["X", "Y"]
    weights = [
        ((x, y), indicator(spatial_corr_func(x, y)))
        for x in range(hallway_length)
        for y in range(hallway_length)
    ]
    joint_dist = TabularDistribution(variables, weights)

    # df = joint_dist.condition({"X":0}).to_df()
    # sns.barplot("Y", "prob", data=df, color="cyan")
    # print(df)
    # plt.ylim(0,1)
    # plt.show()

    agent = HSAgent(hallway_length,
                    None,
                    joint_dist,
                    range_x,
                    range_y,
                    actions=actions)
    # uniform belief over x but true belief over r.
    init_belief_hist = {}
    for state in agent.transition_model.get_all_states():
        if state.r == true_r:
            init_belief_hist[state] = 1.0 / hallway_length
        else:
            init_belief_hist[state] = 0.0
    agent.set_belief(pomdp_py.Histogram(init_belief_hist), prior=True)
    return agent, env


def main(solver="sarsop",
         setting="YR..X",
         spatial_corr_func=spatially_correlated,
         range_x=0, range_y=1,
         viz=True,
         using_jupyter=False,
         vi_options=[],
         vi_reload=False,
         savedir=None,
         num_steps=15,
         actions=ACTIONS):
    pomdp_name = "hallway_search"
    agent, env = initialize_setting(setting,
                                    spatial_corr_func=spatial_corr_func,
                                    range_x=range_x,
                                    range_y=range_y,
                                    actions=actions)
    discount_factor = 0.95

    # Solve the POMDP
    if solver == "sarsop":
        policy = pomdp_py.sarsop(agent, SARSOP_PATH,
                                 discount_factor=discount_factor, timeout=20,
                                 memory=20, precision=1e-12,
                                 pomdp_name=pomdp_name,
                                 remove_generated_files=True)
    elif solver == "vi":
        alphas_path = "%s.alpha" % pomdp_name
        if vi_reload and os.path.exists(alphas_path):
            policy = pomdp_py.AlphaVectorPolicy.construct(alphas_path,
                                                          agent.all_states,
                                                          agent.all_actions,
                                                          solver="vi")
        else:
            policy = pomdp_py.vi_pruning(agent, VI_PATH,
                                         discount_factor=discount_factor,
                                         pomdp_name=pomdp_name,
                                         options=vi_options,
                                         remove_generated_files=True)
    elif solver == "pouct":
        # technically a planner
        policy = pomdp_py.POUCT(max_depth=10, num_sims=2000,
                                discount_factor=discount_factor,
                                exploration_const=100,
                                rollout_policy=agent.policy_model)
    return simulate_policy(policy, agent, env, viz=viz,
                           using_jupyter=using_jupyter, num_steps=num_steps,
                           discount_factor=discount_factor, savedir=savedir)


def simulate_policy(policy, agent, env,
                    num_steps=15,
                    discount_factor=0.95, savedir=None,
                    viz=True, using_jupyter=False):
    if viz:
        if not using_jupyter:
            plt.ion()
            plt.show()
            axes = [plt.gca()]
            if savedir is not None:
                os.makedirs(savedir, exist_ok=True)
        else:
            # In jupyter notebook we plot everything in a single plot.
            _figsize=4
            nrows = int(math.ceil((num_steps+1)//3))
            ncols = 3
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                                     figsize=(_figsize*nrows, _figsize*ncols))
            axes = axes.flatten()
            plt.tight_layout()

        plot_belief(agent.belief, axes[0], env.hallway_length)
        plot_true_state(env.state, axes[0])
        if not using_jupyter:
            plt.show()
            plt.pause(0.001)
            time.sleep(0.5)
            if savedir is not None:
                plt.savefig(os.path.join(savedir, "step-0.png"))

    _cum_reward = 0.0
    _discount = 1.0
    init_belief_hist = copy.deepcopy(agent.belief.get_histogram())
    for step in range(num_steps):
        action = policy.plan(agent)
        reward = env.state_transition(action, execute=True)
        observation = env.true_observation_model.sample(env.state, action)

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

        # Plotting
        if viz:
            if not using_jupyter:
                ax = axes[0]
                ax.clear()
            else:
                ax = axes[step+1]
            plot_belief(agent.belief, ax, env.hallway_length)
            plot_true_state(env.state, ax)
            ax.set_title("({}) action: {}, observation: {}".format(step+1, action, observation))
            if not using_jupyter:
                plt.ion()
                plt.draw()
                plt.pause(0.001)
                time.sleep(0.8)
                if savedir is not None:
                    plt.savefig(os.path.join(savedir, "step-%d.png" % (step+1)))

        # Termination
        if action.name == "Declare":
            print("Done. Discounted cumulative reward: %.5f" % _cum_reward)
            break

    if viz:
        if using_jupyter:
            plt.show()
            if savedir is not None:
                plt.savefig(os.path.join(savedir, "steps.png"))

        plt.clf()
    # Return many things for further analysis
    return (policy, agent, pomdp_py.Histogram(init_belief_hist), _cum_reward)


if __name__ == "__main__":
    simple = "XR.Y"
    simple2 = "Y.XR"
    simple3 = "YXR..."
    range_x, range_y = 0, 1

    # # Value Iteration
    # main(solver="vi", setting=simple2,
    #      spatial_corr_func=spatially_exact,
    #      range_x=range_x, range_y=range_y,
    #      using_jupyter=False, vi_options=['-horizon', 7])

    # SARSOP
    main(solver="sarsop", setting=simple2,
         spatial_corr_func=spatially_exact,
         range_x=range_x, range_y=range_y,
         using_jupyter=False, savedir="examples/%s" % simple2)

    # # POUCT
    # main(solver="pouct", setting=simple2,
    #      spatial_corr_func=spatially_apart,
    #      range_x=range_x, range_y=range_y,
    #      using_jupyter=False)
