# A (PO)MDP agent that can perform navigation on top of
# a THOR environment.

import pomdp_py as pdp
import math
import random

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
        return hash(self.pos, self.rot)

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

    def __init__(self, grid_size=0.25):
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
        forward, angle = action.motion
        x, z = state.pos
        rot = state.rot

        # Because the underlying world is discretized into grids
        # we need to "normalize" the change to x or z to be a
        # scalar of the grid size.
        rot += angle
        dx = forward*math.sin(math.radians(rot))
        dz = forward*math.cos(math.radians(rot))
        x = self.grid_size * round((x + dx) / self.grid_size)
        z = self.grid_size * round((z + dz) / self.grid_size)
        rot = rot % 360
        return NavState((x,z), rot)


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
        return hash(self.pos, self.rot)

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

# Reward model
class NavRewardModel(pdp.RewardModel):
    def __init__(self, goal_pose):
        pos, rot = goal_pose
        self.goal_pose = (pos, round(rot, 2) % 360.0)

    def sample(self, state, action, next_state, **kwargs):
        return self.argmax(state, action, next_state)

    def argmax(self, state, action, next_state, **kwargs):
        next_pose = (next_state.pos, round(next_state.rot, 2) % 360.0)
        if predicted_pose == expected_pose:
            return 100
        else:
            return -1
