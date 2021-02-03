"""
Generalization of the Hallway Search from only X Y domain to
a more customizable one. The initial configuration
can be specified by a string. The sensor configuration,
joint probability distribution, can also be specified
as a string.
"""
import pomdp_py
import random
import numpy as np
import time
import copy
from relpomdp2.constants import SARSOP_PATH, VI_PATH
from relpomdp2.probability import TabularDistribution
import itertools
import matplotlib.pyplot as plt
import seaborn as sns


def ch2id(ch):
    return ord(ch) - ord("A")

def id2ch(objid):
    return chr(ord("A") + objid)

class HSJointState(pomdp_py.SimpleState):
    def __init__(self, r, objlocs):
        """
        True state. Used by environment
        Args:
            r (int): Location of robot
            objlocs (list): location of objects. Index represents object id

        convention: the first object is the target object.
        """
        assert type(objlocs) == tuple
        self.r = r
        self.objlocs = objlocs
        super().__init__((r, objlocs))

    @property
    def x(self):
        """The target object"""
        return self.objlocs[0]

    @property
    def is_terminal(self):
        return self.r == -1

    def to_terminal(self):
        """returns the terminal state"""
        return HSJointState(-1, (-1,)*len(self.objlocs))

    def objloc(self, ref):
        if type(ref) == str:
            return self.objlocs[ch2id(ref)]
        else:
            return self.objlocs[ref]

    def __str__(self):
        return "r%d_%s" % (self.r,
                           "_".join(["obj%d@%d" % (i, self.objlocs[i])
                                    for i in range(len(self.objlocs))]))

class HSState(pomdp_py.SimpleState):
    """simplified state, only includes robot and target"""
    def __init__(self, r, x):
        self.r = r
        self.x = x
        super().__init__((r,x))

    def __str__(self):
        return "r%dx%d" % (self.r, self.x)

    @property
    def is_terminal(self):
        return self.r == -1

    def to_terminal(self):
        """returns the terminal state"""
        return HSState(-1, -1)


class HSAction(pomdp_py.SimpleAction):
    def __init__(self, name):
        super().__init__(name)

class HSObservation(pomdp_py.SimpleObservation):
    def __init__(self, value):
        self.value = value
        super().__init__(value)


######### Transition, Observation, Reward models #######
class HSTransitionModel(pomdp_py.DetTransitionModel):
    def __init__(self, hallway_length):
        self.hallway_length = hallway_length
        super().__init__(epsilon=0.0)

    def r_update(self, r, action):
        if action.name == "left":
            r = r - 1
            r = max(0, min(self.hallway_length-1, r))
        elif action.name == "right":
            r = r + 1
            r = max(0, min(self.hallway_length-1, r))
        return r

    def sample(self, state, action):
        if action.name == "Declare" or state.is_terminal:
            return state.to_terminal()
        else:
            return HSState(self.r_update(state.r, action), state.x)

    def get_all_states(self):
        return [HSState(r, x)
                for r in range(self.hallway_length)
                for x in range(self.hallway_length)]\
                    + [HSState(-1, -1)]  # terminal state

def object_location_combos(num_objects, hallway_length):
    locations = []
    for obj_id in range(num_objects):
        locations.append([l for l in range(hallway_length)])
    all_combos = itertools.product(*locations)
    return all_combos


class HSJointTransitionModel(HSTransitionModel):
    def __init__(self, hallway_length, num_objects):
        self.hallway_length = hallway_length
        self.num_objects = num_objects
        super().__init__(hallway_length)

    def sample(self, state, action):
        if action.name == "Declare" or state.is_terminal:
            return state.to_terminal()
        else:
            return HSJointState(self.r_update(state.r, action),
                                state.objlocs)

    def get_all_states(selff):
        all_combos = object_location_combos(self.num_objects, self.hallway_length)
        return [HSJointState(r, locs)
                for r in range(self.hallway_length)
                for locs in all_combos]\
                    + [HSJointState(-1, (-1,)*self.num_objects)]  # terminal state

def in_range(r, x, range_size):
    return abs(r - x) <= range_size

def indicator(cond):
    if cond:
        return 1.0
    else:
        return 0.0

class HSObservationModel(pomdp_py.ObservationModel):
    def __init__(self, sensor_ranges, sensor_noises):
        """This model assumes the state contains
        all variables. Thus no correlation is needed.

        Args:
            sensor_ranges (map): map from object name (chr) to integer"""
        self.sensor_ranges = sensor_ranges
        self.sensor_noises = sensor_noises
        self._cache = {}  # cache for sample

    def sensor_model(self, r, obj_loc, obj, oval):
        if in_range(r, obj_loc, self.sensor_ranges[obj]):
            if oval == "see-%s" % obj:
                # True positive
                return 1.0 - self.sensor_noises[obj]["FN"]
            elif oval == "nothing-%s" % obj:
                # False negative
                return self.sensor_noises[obj]["FN"]
            else:
                # Invalid observation
                return 0.0
        else:
            if oval == "see-%s" % obj:
                # False positive
                return self.sensor_noises[obj]["FP"]
            elif oval == "nothing-%s" % obj:
                # True negative
                return 1.0 - self.sensor_noises[obj]["FP"]
            else:
                return 0.0

    def probability(self, observation, next_state, action):
        if next_state.is_terminal:
            return indicator(observation.value.startswith("nothing"))

        try:
            if action.name.startswith("look"):
                obj = action.name.split("-")[1]
                obj_loc = next_state.objloc(obj)
                return self.sensor_model(next_state.r, obj_loc, obj, observation.value)
            else:
                return indicator(observation.value.startswith("nothing"))
        except:
            import pdb; pdb.set_trace()

    def sample(self, next_state, action):
        all_observations = self.get_all_observations()
        if (next_state, action) in self._cache:
            probs = self._cache[(next_state, action)]
        else:
            probs = []
            for o in all_observations:
                probs.append(self.probability(o, next_state, action))
            self._cache[(next_state, action)] = probs
        return random.choices(all_observations, weights=probs, k=1)[0]

    def get_all_observations(self):
        """For now, we consider only observation
        of one object at a time, or nothing"""
        all_obs = []
        for obj in sorted(self.sensor_ranges):
            all_obs.append(HSObservation("see-%s" % obj))
            all_obs.append(HSObservation("nothing-%s" % obj))
        return all_obs


class HSRelationObservationModel(HSObservationModel):
    def __init__(self, joint_dist, sensor_ranges, sensor_noises):
        """
        Args:
            joint_dist (Tabulardistribution) a joint distribution that
                that models all variables. (TODO: This is inefficient for now)
            sensor_ranges (list): List of integers of sensor ranges. One for each object.
        """
        self.joint_dist = joint_dist
        self._dist_caches = {}
        super().__init__(sensor_ranges, sensor_noises)

    def corr_model(self, r, x, obj, oval, dist):
        """
        Returns the probility of observing oval (either nothing, or another variable)
        given the location of the robot and the location of x

        Args:
            r (int): robot location
            x (int): target location
            obj (str): character for the object
            oval (str): observation value
            dist (Tabulardistribution): A joint distribution of target and obj_id
        """
        assert ch2id(obj) != 0, "target object observation shouldn't need to use correlation model"
        assert oval == "see-%s" % obj or oval == "nothing-%s" % obj

        cond_dist = dist.condition({id2ch(0):x})
        prob = 0.0
        no_probable_l = True
        for l in cond_dist.ranges[obj]:
            prob += self.sensor_model(r, l, obj, oval) * cond_dist.prob({obj:l})
            if cond_dist.prob({obj:l}) > 0.0:
                no_probable_l = False
        if no_probable_l:
            # There is no where for obj to be so we will definitely see nothing
            return indicator(oval == "nothing-%s" % obj)
        return prob

    def get_pairwise_dist(self, obj1, obj2):
        """Get a pairwise distribution for the given two objects."""
        if (obj1, obj2) in self._dist_caches:
            return self._dist_caches[(obj1, obj2)]
        else:
            return self.joint_dist.sum_out([obj
                                            for obj in self.sensor_ranges
                                            if obj not in {obj1, obj2}])

    def probability(self, observation, next_state, action):
        if next_state.is_terminal:
            return indicator(observation.value.startswith("nothing"))

        if action.name.startswith("look"):
            obj = action.name.split("-")[1]

            if ch2id(obj) == 0:
                return self.sensor_model(next_state.r, next_state.x,
                                         obj, observation.value)
            else:
                if observation.value not in {"see-%s" % obj,
                                             "nothing-%s" % obj}:
                    return 0.0
                else:
                    dist = self.get_pairwise_dist(id2ch(0), obj)  # 0 means target
                    return self.corr_model(next_state.r, next_state.x,
                                            obj, observation.value, dist)
        else:
            return indicator(observation.value.startswith("nothing"))

class HSRewardModel(pomdp_py.DetRewardModel):
    def __init__(self, sensor_costs):
        self.sensor_costs = sensor_costs
    def reward_func(self, state, action, next_state):
        if state.is_terminal:
            return 0.0

        if action.name == "Declare":
            if state.r == state.x:
                return 10
            else:
                return -10
        else:
            if action.name.startswith("look"):
                obj = action.name.split("-")[1]
                return self.sensor_costs[obj]
            else:
                return 0

class HSAgent(pomdp_py.Agent):
    def __init__(self,
                 hallway_length,
                 init_belief,
                 joint_dist,
                 sensor_ranges,
                 sensor_costs,
                 sensor_noises,
                 actions=None):
        if actions is None:
            actions = [HSAction("left"), HSAction("right"),
                       HSAction("Declare")]
            for obj in sorted(sensor_ranges):
                actions.append(HSAction("look-%s" % obj))

        policy_model = pomdp_py.UniformPolicyModel(actions)
        transition_model = HSTransitionModel(hallway_length)
        observation_model = HSRelationObservationModel(joint_dist,
                                                       sensor_ranges, sensor_noises)
        reward_model = HSRewardModel(sensor_costs)
        self.hallway_length = hallway_length
        super().__init__(init_belief,
                         policy_model,
                         transition_model,
                         observation_model,
                         reward_model)

class HSEnv(pomdp_py.Environment):
    def __init__(self,
                 init_state, hallway_length, sensor_costs):
        transition_model = HSJointTransitionModel(hallway_length,
                                                  len(init_state.objlocs))
        reward_model = HSRewardModel(sensor_costs)
        self.hallway_length = hallway_length
        super().__init__(init_state, transition_model, reward_model)

    def print_state(self):
        output = ""
        for i in range(self.hallway_length):
            if self.state.r == i:
                output += "R"
            else:
                added = False
                for obj_id, loc in enumerate(self.state.objlocs):
                    if loc == i:
                        output += id2ch(obj_id)
                        added = True
                        break
                if not added:
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
    ax.bar(xvals, bx, label="$b(A)$")
    ax.set_ylim(0, 1)
    ax.set_xlim(-1, hallway_length)
    ax.set_xticks([i for i in range(hallway_length)])

def plot_true_state(true_state, ax):
    for obj_id in range(len(true_state.objlocs)):
        l = true_state.objlocs[obj_id]
        label = id2ch(obj_id)
        ax.plot([l], 0.05, marker="$%s$" % label,
                color='black', markersize=10, label=label)
    ax.plot([true_state.r], 0.05, "ro", markersize=10, label="robot")
    ax.legend(loc='upper left')

def satisfy_relations(objlocs, pairwise_relations):
    """Returns true if the configuration of objects satisfy the
    pairwise relations"""
    all_true = True
    for obj1, obj2 in pairwise_relations:
        if ch2id(obj1) < len(objlocs) and ch2id(obj2) < len(objlocs):
            spatial_corr_func = pairwise_relations[(obj1, obj2)]
            if not spatial_corr_func(objlocs[ch2id(obj1)], objlocs[ch2id(obj2)]):
                all_true = False
                break
    return all_true

def str_to_setting(situation):
    true_r = None
    objlocs_map = {}  # map from character (object name) to location
    for i, char in enumerate(situation):
        if char == "R":
            true_r = i
        elif char != ".":
            objlocs_map[char] = i
    objlocs = []
    for ch in sorted(objlocs_map):
        objlocs.append(objlocs_map[ch])
        assert id2ch(len(objlocs)-1 == ch), "object characters must be ordered"
    return true_r, tuple(objlocs), len(situation)

def random_setting(numobj, hallway_length,
                   pairwise_relations=None, init_r=None):
    """Returns a string representation of a random setting
    WARNING: the setting generated here DOES NOT match
    the initial belief of the agent. This function is only for convenience!"""
    locations = [l for l in range(hallway_length)]
    if init_r is None:
        init_r = random.sample(locations, 1)[0]
    count = 0
    while count < 1000:
        setting = ["."] * hallway_length
        setting[init_r] = "R"
        objlocs = random.sample(set(locations) - {init_r}, numobj)
        for obj_id in range(numobj):
            setting[objlocs[obj_id]] = id2ch(obj_id)
        if pairwise_relations is not None:
            if satisfy_relations(objlocs, pairwise_relations):
               return "".join(setting)
        else:
            return "".join(setting)
        count += 1
    raise ValueError("Unable to find setting after 1000 attempts.")


def random_init_state(numobj, hallway_length,
                      pairwise_relations=None, init_r=None):
    """The difference between this and random_setting is that,
    random_setting generates a string, which does not allow overlapping
    of objects. But, the domain actually allows overlapping. So,
    this function directly returns the initial state that may
    have such overlap."""
    while True:
        objlocs = []
        for obj_id in range(numobj):
            l = random.sample(set(np.arange(hallway_length)), 1)[0]
            objlocs.append(l)
        if pairwise_relations is not None:
            if satisfy_relations(objlocs, pairwise_relations):
                break
        else:
            break
    if init_r is None:
        init_r = random.sample(set(np.arange(hallway_length)), 1)[0]
    return HSJointState(init_r, tuple(objlocs))

def create_instance(setting_or_init_state,
                    sensor_ranges={"A": 0, "B": 1},
                    sensor_costs={"A": -1, "B": -1},
                    sensor_noises={"A": {"FP":0.0, "FN":0.0},
                                   "B": {"FP":0.0, "FN":0.0}},
                    pairwise_relations={("A","B"):spatially_close},
                    use_correlation_prior=True,
                    actions=None):
    """Given a setting (str) or initial state (HSJointState),
    and other configurations, return an agent and an env.
    The agent's initial belief will be conditioend on the joint
    distribution under the pairwise_relations if `use_correlation_prior`
    is set to True. Otherwise, the initial belief will be uniform
    over the target."""
    if type(setting_or_init_state) == str:
        true_r, objlocs, hallway_length = str_to_setting(setting_or_init_state)
        init_state = HSJointState(true_r, objlocs)
    else:
        init_state, hallway_length = setting_or_init_state
        objlocs = init_state.objlocs
        true_r = init_state.r

    env = HSEnv(init_state, hallway_length, sensor_costs)
    env.true_observation_model = HSObservationModel(sensor_ranges, sensor_noises)

    # Joint probability distribution using pairwise relations
    # (only binary hard spatial correlation for now)
    variables = ["%s" % id2ch(obj_id) for obj_id in range(len(objlocs))]
    objloc_combos = object_location_combos(len(objlocs), hallway_length)
    weights = []
    for objlocs in objloc_combos:
        satisfied = satisfy_relations(objlocs, pairwise_relations)
        weights.append((objlocs, indicator(satisfied)))
    joint_dist = TabularDistribution(variables, weights)

    # Initial belief
    init_belief_dist = {}
    agent = HSAgent(hallway_length,
                    None,
                    joint_dist,
                    sensor_ranges,
                    sensor_costs,
                    sensor_noises,
                    actions=actions)
    # belief over x is uniform, or depends on joint dist, but true belief over r.
    xdist = joint_dist.sum_out(set(joint_dist.variables) - {"A"})
    init_belief_hist = {}
    for state in agent.transition_model.get_all_states():
        if state.r == true_r:
            if use_correlation_prior:
                # the robot considers correlation
                init_belief_hist[state] = xdist.prob({"A":state.x})
            else:
                # the robot does not consider correlation. Uniform prior
                init_belief_hist[state] = 1.0 / hallway_length
        else:
            init_belief_hist[state] = 0.0
    agent.set_belief(pomdp_py.Histogram(init_belief_hist), prior=True)
    return agent, env

def compute_policy(agent, env, pomdp_name="hallway_search",
                   discount_factor=0.95,
                   solver="sarsop",
                   **solver_params):
    _start_time = time.time()
    if solver == "sarsop":
        policy = pomdp_py.sarsop(
            agent, SARSOP_PATH,
            discount_factor=discount_factor,
            timeout=solver_params.get("timeout", 20),
            memory=solver_params.get("memory", 20),
            precision=solver_params.get("precision", 1e-12),
            pomdp_name=pomdp_name,
            remove_generated_files=solver_params.get("remove_generated_files", True),
            logfile=solver_params.get("logfile", "%s_sarsop.log" % pomdp_name))
    elif solver == "pouct":
        # This of course, does not explicitly solve. It just creates an online planner
        policy = pomdp_py.POUCT(max_depth=solver_params.get("max_depth", 30),
                                num_sims=solver_params.get("num_sims", 2000),
                                discount_factor=discount_factor,
                                exploration_const=solver_params.get("exploration_const", 100),
                                rollout_policy=agent.policy_model)
    solver_time = time.time() - _start_time
    return policy, solver_time

def simulate_policy(policy, agent, env, viz=False,
                    num_steps=100, discount_factor=0.95, hardcode_plan=None):
    _history = [(copy.deepcopy(env.state), None, None, None)]  # Stores [(s,a,o,r), ...]
    _meta = {}

    if viz:
        plt.ion()
        plt.show()
        axes = [plt.gca()]
        plot_belief(agent.belief, axes[0], env.hallway_length)
        plot_true_state(env.state, axes[0])
        plt.show()
        plt.pause(0.001)
        time.sleep(0.5)

    _cum_reward = 0.0
    _discount = 1.0
    _odiff_count = 0
    for step in range(num_steps):
        if hardcode_plan is not None and step < len(hardcode_plan):
            action = hardcode_plan[step]
        else:
            _start_time = time.time()
            action = policy.plan(agent)
            _meta["solver_time"] = _meta.get("solver_time", 0.0) + (time.time() - _start_time)

        reward = env.state_transition(action, execute=True)
        observation = env.true_observation_model.sample(env.state, action)
        observation_expected = agent.observation_model.sample(
            HSState(env.state.r, env.state.x), action)
        if not(observation == observation_expected) and action.name.startswith("look"):
            _odiff_count += 1

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

        _history.append((copy.deepcopy(env.state), action, observation, reward))

        if viz:
            axes[0].clear()
            plot_belief(agent.belief, axes[0], env.hallway_length)
            plot_true_state(env.state, axes[0])
            axes[0].set_title("({}) action: {}, observation: {}".format(step+1, action,
                                                                        observation))
            plt.draw()
            plt.pause(0.001)
            time.sleep(0.8)

        if action.name == "Declare":
            print("Done.")
            break
    _meta["odiff_count"] = _odiff_count
    return _cum_reward, _history, _meta


def end2end(setting_or_init_state_or_genparams,
            solver="sarsop",
            discount_factor=0.95,
            instance_config={},
            solver_params={},
            hardcode_plan=None):
    """End to end run an instance

    instance_config include:
    - all parameters in `create_istance`"""
    setting_or_init_state = setting_or_init_state_or_genparams
    if type(setting_or_init_state_or_genparams) == tuple:
        numobj, hallway_length = setting_or_init_state_or_genparams[:2]
        init_r = None
        if len(setting_or_init_state_or_genparams) == 3:
            init_r = setting_or_init_state_or_genparams[2]
        setting_or_init_state = random_init_state(numobj, hallway_length,
                                                  pairwise_relations=pairwise_relations,
                                                  init_r=init_r)
    agent, env = create_instance(setting_or_init_state,
                                 **instance_config)
    policy, solver_time = compute_policy(agent, env,
                                         solver=solver,
                                         discount_factor=discount_factor,
                                         **solver_params)
    cum_reward, _history, _meta = simulate_policy(policy, agent, env, viz=True,
                                                  discount_factor=discount_factor,
                                                  hardcode_plan=hardcode_plan)
    return cum_reward


if __name__ == "__main__":
    end2end("RAB..",
            solver="sarsop",
            instance_config=dict(
                sensor_ranges={"A": 0, "B": 2},#, "C": 3},
                sensor_costs={"A": -1, "B": 0, "C": 0},
                sensor_noises={"A": {"FP":0.0, "FN":0.0},
                               "B": {"FP":0.0, "FN":0.0},
                               "C": {"FP":0.0, "FN":0.0}},
                pairwise_relations={("A","B"):spatially_independent,
                                    ("B","C"):spatially_independent},
            ))
