# Object search in THOR
# This code doesn't use the oo-pomdp framework, for simplicity.
# Also, this code considers 2d robot state space,
# and currently a 2d target state space. TODO: extend the latter to 3D.
# The observation can still be volumetric..

import pomdp_py as pdp
import random
import math
from relpomdp.realistic.environment import ThorEnv
from relpomdp.realistic.utils.ai2thor_utils import save_frames,\
    plot_reachable_grid, get_reachable_pos_set, scene_info, visible_objects, save_frames
from relpomdp.realistic.utils.util import euclidean_dist, in_range_inclusive
from relpomdp.home2d.domain.visual import lighter
from pprint import pprint
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

class State(pdp.State):
    """A state is factored into robot state and the target object state"""
    def __init__(self, robot_state, target_state):
        self.robot_state = robot_state
        self.target_state = target_state
        self._hashcode = hash((self.robot_state, self.target_state))

    def __str__(self):
        return "state({}, {})".format(self.robot_state, self.target_state)

    def __repr__(self):
        return str(self)

    @property
    def robot_pose(self):
        return self.robot_state.pose

    def __eq__(self, other):
        if isinstance(other, State):
            return self.target_state == other.target_state\
                and self.robot_state == other.robot_state
        return False

    def __hash__(self):
        return self._hashcode

class RobotState(pdp.State):
    def __init__(self, pose, found):
        self.pose = pose
        self.found = found
        self._hashcode = hash((self.pose, self.found))

    def __str__(self):
        return "Robot({}, {})".format(self.pose, self.found)

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        if isinstance(other, RobotState):
            return self.pose == other.pose\
                and self.found == other.found
        return False

    def __hash__(self):
        return self._hashcode

class TargetState(pdp.State):
    def __init__(self, objclass, pose):
        self.objclass = objclass
        self.pose = pose
        self._hashcode = hash((self.objclass, self.pose))

    def __str__(self):
        return "Target({}, {})".format(self.objclass, self.pose)

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        if isinstance(other, TargetState):
            return self.objclass == other.objclass\
                and self.pose == other.pose
        return False

    def __hash__(self):
        return self._hashcode


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
        motion (tuple): a (vt, vw) tuple for translational, rotational velocities
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
            # MotionAction("MoveBack", (-grid_size, 0)),
            MotionAction("RotateLeft", (0, -degrees)),
            MotionAction("RotateRight", (0, degrees))}

# Right now, just a simple action
class DeclareFound(Action):
    def __init__(self):
        super().__init__("declare-found")


# Transition model
def motion_model(pose, motion, grid_size=0.25):
    """
    Returns the next pose by applying the motion.

    Known issues:
    * When the action's rotation is something other than 90 degrees,
      this model doesn't always predict the correct next agent pose
      in THOR. 45 degrees has fewer errors than 30 degrees. The error
      is usually off-by-one grid cell. For this reason, if you set
      the action to be non-90 rotation, you may want to force THOR
      to teleoperate the agent to the sampled pose.

    Args:
        pose (tuple): a tuple ((x,z), th)
        motion (tuple): a tuple (vt, vw)
    """
    forward, angle = motion
    x, z = pose[0]  # position
    rot = pose[1]   # rotation

    # Because the underlying world is discretized into grids
    # we need to "normalize" the change to x or z to be a
    # scalar of the grid size.
    rot += angle
    dx = forward*math.sin(math.radians(rot))
    dz = forward*math.cos(math.radians(rot))
    x = grid_size * round((x + dx) / grid_size)
    z = grid_size * round((z + dz) / grid_size)
    rot = rot % 360
    return ((x,z), rot)

class TransitionModel(pdp.TransitionModel):
    def __init__(self, grid_size=0.25):
        self.grid_size = grid_size

    def sample(self, state, action):
        next_target_state = TargetState(state.target_state.objclass,
                                       state.target_state.pose)
        if isinstance(action, MotionAction):
            next_pose = motion_model(state.robot_pose,
                                     action.motion,
                                     grid_size=self.grid_size)
            next_state = State(RobotState(next_pose, state.robot_state.found),
                               next_target_state)

        elif isinstance(action, DeclareFound):
            next_state = State(RobotState(state.robot_pose, True),
                               next_target_state)

        else:
            raise ValueError("Invalid action {}".format(action))

        return next_state

# Observation
class PoseObservation(pdp.Observation):
    def __init__(self, pose):
        self.pose = pose

    def __str__(self):
        return "{}({})".format(self.__class__.__name__,
                               self.pose)

    def __hash__(self):
        return hash(self.pose)

    def __eq__(self, other):
        if isinstance(other, Observation):
            return other.pose == self.pose
        return False

class RobotObservation(PoseObservation):
    def __init__(self, pose, found):
        self.found = found
        super().__init__(pose)

class TargetObservation(PoseObservation):
    def __init__(self, objclass, pose):
        self.objclass = objclass
        super().__init__(pose)

class Observation(pdp.Observation):
    def __init__(self, robot_observation, target_observation):
        self.robot_observation = robot_observation
        self.target_observation = target_observation
        self._hash_code = hash((self.robot_observation, self.target_observation))

    def __hash__(self):
        return self._hash_code

    def __eq__(self, other):
        if isinstance(other, Observation):
            return other.robot_observation == self.robot_observation\
                and other.target_observation == self.target_observation
        return False


# Observation Model
class FanSensor:
    """2D fanshape sensor"""
    def __init__(self, fov=90, min_range=0.01, max_range=0.50):
        self.fov = math.radians(fov % 360)  # store fov in radians
        self.min_range = min_range
        self.max_range = max_range

    def within_range(self, robot_pose, point):
        if robot_pose[0] == point:
            return True

        (x, z), th = robot_pose
        dist = euclidean_dist((x,z), point)
        if self.min_range <= dist <= self.max_range:
            th_point = math.atan2(point[0] - x, point[1] - z)
            bearing = (th_point - math.radians(th))  % (2*math.pi)

            # because we defined bearing to be within 0 to 360, the fov
            # angles should also be defined within the same range.
            fov_ranges = (0, self.fov/2), (2*math.pi - self.fov/2, 2*math.pi)
            if in_range_inclusive(bearing, fov_ranges[0])\
               or in_range_inclusive(bearing, fov_ranges[1]):
                return True
            else:
                return False
        return False

class SensorObservationModel(pdp.ObservationModel):
    def __init__(self, sensor, objclass, detection_prob):
        self.sensor = sensor
        self.objclass = objclass
        self.detection_prob = detection_prob

    def sample(self, next_state, action):
        robot_observation = RobotObservation(next_state.robot_pose,
                                             next_state.robot_state.found)
        in_fov = self.sensor.within_range(next_state.robot_pose,
                                          next_state.target_state.pose)
        if in_fov:
            if random.uniform(0, self.detection_prob) < 1.0:
                target_observation = TargetObservation(self.objclass, next_state.target_state.pose)
            else:
                target_observation = TargetObservation(self.objclass, None)
        else:
            target_observation = TargetObservation(self.objclass, None)
        return target_observation

    def probability(self, target_observation, next_state, action):
        in_fov = self.sensor.within_range(next_state.robot_pose,
                                              next_state.target_state.pose)
        if in_fov:
            if target_observation.pose == next_state.target_state.pose:
                return self.detection_prob
            else:
                return 1. - self.detection_prob
        else:
            if target_observation.pose == None:
                return 1.0 - 1e-9
            else:
                return 1e-9

    def visualize(self, ax, robot_pose, reachable_positions):
        """Plot the field of view"""
        x, z = [], []
        for pos in reachable_positions:
            if self.sensor.within_range(robot_pose, pos):
                x.append(pos[0])
                z.append(pos[1])
        ax.scatter(x, z, s=120.0, c="#fff957", alpha=0.7)

# Reward Model
class RewardModel(pdp.RewardModel):
    def __init__(self, sensor):
        self.sensor = sensor

    def sample(self, state, action, next_state):
        # Check if the robot is facing the target
        if isinstance(action, DeclareFound):
            assert next_state.robot_state.found == True

            robot_pos, robot_rot = next_state.robot_pose
            target_pos = next_state.target_state.pose

            if self.sensor.within_range(next_state.robot_pose,
                                        target_pos):
                return 100.0
            return -100.0
        else:
            return -1.0

# rollout Policy Model
class PolicyModel(pdp.RolloutPolicy):
    def __init__(self, actions, reachable_positions, grid_size=0.25):
        self.actions = actions
        self.reachable_positions = reachable_positions
        self.grid_size = grid_size

        self._translation_actions = {a for a in self.actions
                                     if isinstance(a, MotionAction) and a.motion[0] != 0}
        self._rotation_actions = {a for a in self.actions
                                  if isinstance(a, MotionAction) and a.motion[1] != 0}

        # Make sure the only other kind of action is DeclareFound
        assert actions == self._translation_actions | self._rotation_actions | {DeclareFound()}

    def sample(self, state, **kwargs):
        return random.sample(self.get_all_actions(state=state), 1)[0]

    def rollout(self, state, history=None):
        return random.sample(self.get_all_actions(state=state), 1)[0]

    def get_all_actions(self, state=None, history=None):
        if state is None:
            return self.actions
        else:
            # Prune translation actions that lead to unreachable positions.
            all_actions = self._rotation_actions | {DeclareFound()}
            for action in self._translation_actions:
                next_pose = motion_model(state.robot_pose, action.motion,
                                         grid_size=self.grid_size)
                if next_pose[0] in self.reachable_positions:
                    all_actions.add(action)
            return all_actions

# Belief
class Belief(pdp.Histogram):
    def __init__(self, target_class, hist):
        self.target_class = target_class
        super().__init__(hist)

    @classmethod
    def uniform(cls,
                init_robot_pose, target_class,
                possible_positions):
        hist = {}
        total = 0.0
        for position in possible_positions:
            robot_state = RobotState(init_robot_pose, False)
            target_state = TargetState(target_class, position)
            state = State(robot_state, target_state)
            hist[state] = 1.0
            total += hist[state]
        for state in hist:
            hist[state] /= total
        return Belief(target_class, hist)

    def update(self, action, observation, sensor_model):
        next_robot_state = RobotState(observation.robot_observation.pose,
                                      observation.robot_observation.found)

        possible_positions = set()
        hist = {}
        for state in self:
            next_target_state = TargetState(state.target_state.objclass,
                                            state.target_state.pose)
            possible_positions.add(next_target_state.pose)
            next_state = State(next_robot_state, next_target_state)

            target_observation = observation.target_observation
            hist[next_state] = self[state]\
                               * sensor_model.probability(target_observation,
                                                          next_state, action)
        return Belief(self.target_class, hist)

    def visualize(self, ax, color=(128, 128, 128)):
        hist = self.get_histogram()
        colors = []
        x = []
        z = []
        last_val = -1
        for state in reversed(sorted(hist, key=hist.get)):
            if last_val != -1:
                color = lighter(color, 1-hist[state]/last_val)
            if np.mean(np.array(color) / np.array([255, 255, 255])) < 0.999:
                tx, tz = state.target_state.pose
                x.append(tx)
                z.append(tz)
                colors.append(color)
                last_val = hist[state]
                if last_val <= 0:
                    break
        ax.scatter(x, z, c=np.array(colors)/255.0, s=30.0)

        robot_pose = self.mpe().robot_pose
        plot_robot(ax, robot_pose)
        ax.set_xlabel("X axis")
        ax.set_ylabel("Z axis")


def plot_robot(ax, robot_pose, color='b'):
    pos, rot = robot_pose
    ax.scatter([pos[0]], [pos[1]], c=color)
    ax.arrow(pos[0], pos[1],
             0.2*math.sin(math.radians(rot)),  # dx
             0.2*math.cos(math.radians(rot)),  # dz
             width=0.005, head_width=0.05, color=color)


def possible_positions(reachable_positions, grid_size=0.25):
    """Obtain the possible possitions where an object
    can be located given the environment and reachable positions."""
    positions = set(reachable_positions)
    for x, z in reachable_positions:
        positions.add((x+grid_size, z))
        positions.add((x-grid_size, z))
        positions.add((x, z+grid_size))
        positions.add((x, z-grid_size))
        positions.add((x+grid_size, z+grid_size))
        positions.add((x-grid_size, z-grid_size))
    return positions


# Test System.
def find_closest(q, points):
    """Given a 2d point and a list of 2d points,
    return the point in `points` that's closest to `q`
    by euclidean_dist"""
    return min(points,
               key=lambda p: euclidean_dist(p, q))

def plot_step(ax, fig, env, belief,
              robot_pose, reachable_positions, sensor_model):
    robot_pose = belief.mpe().robot_pose

    ax.clear()
    plot_reachable_grid(env.controller, ax,
                        agent_pose=env.agent_pose(), s=80.0)
    belief.visualize(ax)
    sensor_model.visualize(ax, robot_pose, reachable_positions)
    fig.canvas.draw()
    fig.canvas.flush_events()

def save_frame(savepath, img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(savepath, img)

def test_system(scene_name, grid_size=0.25, degrees=90):
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
    # env.controller.step(action="ToggleMapView")
    event = env.controller.step(action="Pass")
    reachable_positions = get_reachable_pos_set(env.controller, use_2d=True)

    # problem instance
    init_pose = env.agent_pose(use_2d=True)

    sinfo = scene_info(env.controller.step(action="Pass").metadata)
    target_options = set(sinfo["TypeCount"].keys())
    pprint(sinfo)
    while True:
        target_class = input("Enter a target class from the list above: ")
        if target_class in target_options:
            break
        else:
            print("{} is invalid. Try again.".format(target_class))
    # target_class = "Box"

    init_belief = Belief.uniform(init_pose, target_class,
                                 possible_positions(reachable_positions, grid_size=grid_size))
    transition_model = TransitionModel(grid_size=grid_size)
    sensor = FanSensor(fov=90, min_range=0.0, max_range=grid_size*2)
    sensor_model = SensorObservationModel(sensor, target_class, 0.9)
    reward_model = RewardModel(sensor)
    policy_model = PolicyModel(motions | {DeclareFound()},
                               reachable_positions, grid_size=grid_size)

    # environment state: It's not necessary to query THOR to obtain the true
    # target location for POMDP planning or belief update, because targets are
    # static. Therefore pose is set to None.
    state = State(init_belief.mpe().robot_state,
                  TargetState(target_class, None))

    agent = pdp.Agent(init_belief,
                      policy_model,
                      transition_model,
                      sensor_model,
                      reward_model)
    planner = pdp.POUCT(max_depth=50,
                        discount_factor=0.95,
                        num_sims=1000,
                        exploration_const=100,
                        rollout_policy=agent.policy_model)

    # frame saving
    savepath = "frames/%s-%s" % (scene_name, target_class)
    os.makedirs(savepath, exist_ok=True)
    save_frame(os.path.join(savepath, "frame-0.png"), event.frame)

    # plotting
    plt.ion()
    fig = plt.figure(figsize=(6,3))
    ax = fig.add_subplot(1, 1, 1)#, projection="3d")
    plt.show(block=False)
    plot_step(ax, fig, env, init_belief,
              init_pose, reachable_positions, sensor_model)
    plt.savefig(os.path.join(savepath, "belief-0.png"))

    # Start search
    target_found = False
    for step in range(100):
        action = planner.plan(agent)
        agent.tree.print_children_value()

        # Execute move action, receive THOR event
        next_state = transition_model.sample(state, action)
        assert next_state.robot_pose[0] in reachable_positions
        if degrees != 90:
            print("Warning: rotation degree isn't 90, Ai2thor motion model"\
                  "doesn't exactly match my transition model.")
        if isinstance(action, MotionAction):
            thor_action_name, params = action.to_thor_action()
            event = env.controller.step(action=thor_action_name, **params)
        else:
            event = env.controller.step(action="Pass")

        # Process event to get observation
        robot_pose = env.agent_pose(use_2d=True)

        # Check whether there is an instance of target class that is
        # visible; If so, obtain the pose in the reachable positions
        # closest to the target pose, as the observation.
        # NOTE: This observation comes directly from the THOR environment.
        # Its object poses are therefore groundtruth. So it's not realistic.
        # Also, receiving such observation ignores the sensor's range!
        objects = visible_objects(event.metadata)
        detected_pos = None
        for obj in objects:
            if obj['objectType'] == target_class:
                obj_pos = (obj['position']['x'], obj['position']['z'])
                if sensor.within_range(robot_pose, obj_pos):
                    # The THOR's frame may show more stuff than what the sensor
                    # can really detect.
                    detected_pos = find_closest(obj_pos,
                                                reachable_positions)
                    if isinstance(action, DeclareFound):
                        target_found = True
                    break
        target_observation = TargetObservation(target_class, detected_pos)
        observation = Observation(RobotObservation(robot_pose, next_state.robot_state.found),
                                  target_observation)

        print("Step {} | Action: {} | Detection: {} | Belief MPE: {}"\
              .format(step, action, target_observation, agent.belief.mpe()))

        if isinstance(action, DeclareFound):
            print("Done. Target actually found? ", target_found)
            break

        # Update belief
        new_belief = agent.belief.update(action, observation, sensor_model)
        agent.set_belief(new_belief)
        planner.update(agent, action, observation)

        # Apply state transition
        state = next_state

        plot_step(ax, fig, env, agent.belief,
                  robot_pose, reachable_positions, sensor_model)

        # saving frames
        plt.savefig(os.path.join(savepath, "belief-%d.png" % (step+1)))
        save_frame(os.path.join(savepath, "frame-%d.png" % (step+1)), event.frame)

if __name__ == "__main__":
    test_system("FloorPlan28", grid_size=0.50)
