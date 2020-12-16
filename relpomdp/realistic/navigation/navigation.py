# A (PO)MDP agent that can perform navigation on top of
# a THOR environment.

import pomdp_py as pdp
import math
import random
from relpomdp.realistic.environment import ThorEnv
from relpomdp.realistic.utils.ai2thor_utils import save_frames,\
    plot_reachable_grid, get_reachable_pos_set
from relpomdp.realistic.object_search.object_search import motion_model
import matplotlib.pyplot as plt

# State
class NavState(pdp.State):
    def __init__(self, pos, rot):
        """
        pos: 2d position of robot, (x,z) in Unity coordinate system
        rot: a float for the rotation around y axis (vertical axis), in degrees
        """
        self.pos = pos
        self.rot = rot

    def __str__(self):
        return '%s::(%s, %s)' % (str(self.__class__.__name__),
                                str(self.pos),
                                str(self.rot))

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        if isinstance(other, NavState):
            return other.pos == self.pos\
                and other.rot == self.rot
        return False

    def __hash__(self):
        return hash((self.pos, self.rot))

# Action
class Action(pdp.Action):
    """Mos action; Simple named action."""
    def __init__(self, name):
        self.name = name
    def __hash__(self):
        return hash(self.name)
    def __eq__(self, other):
        if isinstance(other, Action):
            return self.name == other.name
        elif type(other) == str:
            return self.name == other
    def __str__(self):
        return self.name
    def __repr__(self):
        return "Action(%s)" % self.name

class MotionAction(Action):
    def __init__(self, motion_name, motion):
        """
        motion (tuple): a (vt, vw) tuple for translationa, rotational velocities
            vt is in meters, vw is in degrees.
        """
        self.motion = motion
        super().__init__(motion_name)

    def to_thor_action(self):
        """Returns an action representation in Thor
        as a tuple, (action_name:str, params:dict)"""
        vt, vw = self.motion
        if vt != 0 and vw != 0:
            raise ValueError("Ai2Thor does not support actions"\
                             "to change transition and rotation together.")
        if vt != 0:
            return (self.name, {"moveMagnitude": abs(vt)})
        else:
            return (self.name, {"degrees": abs(vw)})


def build_motion_actions(grid_size=0.25, degrees=45):
    # Ai2Thor motion actions
    return {MotionAction("MoveAhead", (grid_size, 0)),
            MotionAction("MoveBack", (-grid_size, 0)),
            MotionAction("RotateLeft", (0, -degrees)),
            MotionAction("RotateRight", (0, degrees))}

# Observation
class NavObservation(pdp.Observation):
    def __repr__(self, pos, rot):
        self.pos = pos
        self.rot = rot

    def __str__(self):
        return '%s::(%s,%s)' % (str(self.__class__.__name__),
                                str(self.pos),
                                str(self.rot))

    def __hash__(self):
        return hash(self.pos, self.rot)

    def __eq__(self, other):
        if isinstance(other, NavObservation):
            return other.pos == self.pos\
                and other.rot == self.rot
        return False


SIGN = lambda x: -1.0 if x < 0.0 else 1.0

# Transition model
class TransitionModel(pdp.TransitionModel):

    def __init__(self, reachable_positions, grid_size=0.25):
        self.reachable_positions = reachable_positions
        self.grid_size = grid_size

    def sample(self, state, action):
        """
        A simple 2D transition model.

        Known issues:
        * When the action's rotation is something other than 90 degrees,
          this model doesn't always predict the correct next agent pose
          in THOR. 45 degrees has fewer errors than 30 degrees. The error
          is usually off-by-one grid cell. For this reason, if you set
          the action to be non-90 rotation, you may want to force THOR
          to teleoperate the agent to the sampled pose.
        """
        next_pose = motion_model((state.pos, state.rot), action.motion)
        # forward, angle = action.motion
        # x, z = state.pos
        # rot = state.rot

        # # Because the underlying world is discretized into grids
        # # we need to "normalize" the change to x or z to be a
        # # scalar of the grid size.
        # rot += angle
        # dx = forward*math.sin(math.radians(rot))
        # dz = forward*math.cos(math.radians(rot))
        # x = self.grid_size * round((x + dx) / self.grid_size)
        # z = self.grid_size * round((z + dz) / self.grid_size)
        # rot = rot % 360
        (x,z), rot = next_pose
        if (x,z) in self.reachable_positions:
            return NavState((x,z), rot)
        else:
            return NavState(state.pos, state.rot)


# Observation
class NavObservation(pdp.Observation):
    def __init__(self, pos, rot):
        """
        pos: 2d position of robot, (x,z) in Unity coordinate system
        rot: a float for the rotation around y axis (vertical axis), in degrees
        """
        self.pos = pos
        self.rot = rot

    def __str__(self):
        return '%s::(%s, %s)' % (str(self.__class__.__name__),
                                str(self.pos),
                                str(self.rot))

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        if isinstance(other, NavObservation):
            return other.pos == self.pos\
                and other.rot == self.rot
        return False

    def __hash__(self):
        return hash((self.pos, self.rot))

# Observation model: A noiseless observation model
class ObservationModel(pdp.ObservationModel):
    def __init__(self):
        pass

    def sample(self, next_state, action):
        return NavObservation(next_state.pos, next_state.rot)

# Policy model
class RandomPolicyModel(pdp.RolloutPolicy):
    def __init__(self, actions):
        self.actions = actions

    def sample(self, state, **kwargs):
        return random.sample(self.actions, 1)[0]

    def rollout(self, state, history=None):
        return self.sample(state)

    def get_all_actions(self, state=None, history=None):
        return self.actions

# Reward model
class NavRewardModel(pdp.RewardModel):
    def __init__(self, goal_pos):
        self.goal_pos = goal_pos

    def sample(self, state, action, next_state, **kwargs):
        return self.argmax(state, action, next_state)

    def argmax(self, state, action, next_state, **kwargs):
        if next_state.pos == self.goal_pos:
            return 100
        else:
            return -1

# Agent
class NavAgent(pdp.Agent):
    def __init__(self,
                 init_pose, goal_pose, actions, reachable_positions,
                 nparticles=100, grid_size=0.25):
        init_belief = pdp.Particles([NavState(*init_pose) for i in range(nparticles)])
        super().__init__(init_belief,
                         RandomPolicyModel(actions),
                         TransitionModel(reachable_positions, grid_size),
                         ObservationModel(),
                         NavRewardModel(goal_pose))

# Test
def test(scene_name, nparticles=1000, grid_size=0.25, degrees=90):
    config = {
        "scene_name": scene_name,
        "agent_mode": "default",
        "width": 400,
        "height": 400,
        "grid_size": grid_size
    }
    motions = build_motion_actions(grid_size=grid_size, degrees=degrees)

    env = ThorEnv(config)
    env.launch()
    reachable_positions = get_reachable_pos_set(env.controller, use_2d=True)

    # plotting
    plt.ion()
    fig = plt.figure(figsize=(6,3))
    ax = fig.add_subplot(1, 1, 1)#, projection="3d")
    plt.show(block=False)

    # problem instance
    init_pose = env.agent_pose(use_2d=True)
    goal_pos = random.sample(reachable_positions, 1)[0]
    init_y = env.agent_pose()[0][1]  # Note: needed due to Ai2thor bug
    print("Init pose: {}     Goal pose: {}".format(init_pose, goal_pos))

    agent = NavAgent(init_pose, goal_pos, motions, reachable_positions,
                     nparticles=100,
                     grid_size=grid_size)
    planner = pdp.POMCP(max_depth=100,
                        discount_factor=0.95,
                        num_sims=2000,
                        exploration_const=100,
                        rollout_policy=agent.policy_model)

    # Environment
    init_state = NavState(*init_pose)
    transition_model = agent.transition_model
    reward_model = agent.reward_model
    observation_model = agent.observation_model
    state = init_state

    for step in range(100):
        action = planner.plan(agent)
        next_state = transition_model.sample(state, action)

        # Check whether next_state should be applied
        if next_state.pos not in reachable_positions:
            # next state pos is invalid
            next_state = state  # no state transition
        else:
            # Yes, robot can move to next state
            if degrees != 90:
                print("Warning: rotation degree isn't 90, Ai2thor motion model"\
                      "doesn't exactly match my transition model.")
            thor_action_name, params = action.to_thor_action()
            event = env.controller.step(action=thor_action_name, **params)

        observation = observation_model.sample(next_state, action)
        reward = reward_model.sample(state, action, next_state)

        plot_reachable_grid(env.controller, ax, agent_pose=env.agent_pose())
        ax.scatter([goal_pos[0]], [goal_pos[1]], c='g')
        fig.canvas.draw()
        fig.canvas.flush_events()

        # belief update
        try:
            planner.update(agent, action, observation)
        except ValueError:
            true_pose = env.agent_pose(use_2d=True)
            agent.set_belief(pdp.Particles([NavState(*true_pose)
                                            for i in range(nparticles)]))
        print("Step {} | Reward: {} | Action: {} | Observation: {} | Belief MPE: {}"\
              .format(step, reward, action, observation, agent.belief.mpe()))

        state = next_state
        if state.pos == goal_pos:
            print("Done.")
            break


if __name__ == "__main__":
    test("FloorPlan_Train1_1", nparticles=1000, grid_size=0.75)
