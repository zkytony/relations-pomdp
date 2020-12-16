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
    plot_reachable_grid, get_reachable_pos_set, scene_info, visible_objects
from relpomdp.realistic.utils.util import euclidean_dist
from relpomdp.realistic.object_search.sensor import FanSensor
from pprint import pprint

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
            MotionAction("MoveBack", (-grid_size, 0)),
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

    def __str__(self):
        return "{}({})".format(self.__class__.__name__,
                               self.pose)

    def __hash__(self):
        return self._hash_code

    def __eq__(self, other):
        if isinstance(other, Observation):
            return other.robot_observation == self.robot_observation\
                and other.target_observation == self.target_observation
        return False


# Observation Model
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
            if random.uniform(0, self.detection_prob):
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
            if target_observation.pose is None:
                return 1 - self.detection_prob
            else:
                assert target_observation.pose == next_state.target_state.pose
                return self.detection_prob
        else:
            if target_observation.pose is None:
                return 1.0 - 1e-9
            else:
                return 1e-9

# Reward Model
class RewardModel(pdp.RewardModel):
    def __init__(self, declare_range=1.0):
        self.declare_range = declare_range

    def sample(self, state, action, next_state):
        # Check if the robot is facing the target
        if isinstance(action, DeclareFound):
            assert next_state.robot_state.found == True
            # (robot_x, robot_z), robot_rot = next_state.robot_pose
            # target_pos = next_state.target_state.pose

            # # dot product between the robot's forward direction
            # # and the vector to the target
            # forward_vector = (robot_x + math.sin(robot_rot),
            #                   robot_z + math.cos(robot_rot))
            # target_vector = (target_pos[0] - robot_x,
            #                  target_pos[1] - robot_z)
            # if forward_vector[0]*target_vector[0]\
            #    + forward_vector[1]*target_vector[1] > 0:
            #     # same direction
            #     if euclidean_dist(target_pos, (robot_x, robot_z)) <= self.declare_range:
            #         return 100
            if next_state.robot_state.pose[0] == next_state.target_state.pose:
                return 100.0
            else:
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
                if next_pose in self.reachable_positions:
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

# Test
def find_closest(q, points):
    """Given a 2d point and a list of 2d points,
    return the point in `points` that's closest to `q`
    by euclidean_dist"""
    return min(points,
               key=lambda p: euclidean_dist(p, q))


def test(scene_name, grid_size=0.25, degrees=90):
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

    init_belief = Belief.uniform(init_pose, target_class, reachable_positions)
    transition_model = TransitionModel(grid_size=grid_size)
    sensor = FanSensor(fov=90, min_range=0.0, max_range=grid_size*2)
    sensor_model = SensorObservationModel(sensor, target_class, 0.9)
    reward_model = RewardModel(declare_range=grid_size*2)
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

    target_found = False
    for step in range(100):
        action = planner.plan(agent)

        # Execute move action, receive THOR event
        next_state = transition_model.sample(state, action)
        assert next_state.robot_pose[0] in reachable_positions
        if degrees != 90:
            print("Warning: rotation degree isn't 90, Ai2thor motion model"\
                  "doesn't exactly match my transition model.")
        thor_action_name, params = action.to_thor_action()
        event = env.controller.step(action=thor_action_name, **params)

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
                detected_pos = find_closest(obj_pos,
                                            reachable_positions)
                if isinstance(action, DeclareFound):
                    target_found = True
                break

        if isinstance(action, DeclareFound):
            print("Done. Target actually found? ", target_found)
            break

        observation = Observation(RobotObservation(robot_pose, next_state.robot_state.found),
                                  TargetObservation(target_class, detected_pos))
        new_belief = agent.belief.update(action, observation, sensor_model)
        agent.set_belief(new_belief)
        planner.update(agent, action, observation)

if __name__ == "__main__":
    test("FloorPlan_Train1_1", grid_size=0.5)
