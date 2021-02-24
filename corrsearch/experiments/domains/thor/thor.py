"""
Functions related to THOR simulation
"""
import pomdp_py
import numpy as np
import math
from ai2thor.controller import Controller
from corrsearch.experiments.domains.thor.grid_map import GridMap
from corrsearch.experiments.domains.thor.transition import DetRobotTrans
from corrsearch.utils import remap, to_rad, euclidean_dist
from corrsearch.models.transition import *
from corrsearch.models.robot_model import *
from corrsearch.models.problem import *
from corrsearch.models.state import *

def reachable_thor_loc2d(controller):
    """
    Returns a tuple (x, z) where x and z are lists corresponding to x/z coordinates.
    You can obtain a set of 2d positions tuples by:
        `set(zip(x, z))`
    """
    # get reachable positions
    event = controller.step(action="GetReachablePositions")
    positions = event.metadata["actionReturn"]
    x = np.array([p['x'] for p in positions])
    y = np.array([p['y'] for p in positions])
    z = np.array([p['z'] for p in positions])
    return x, z

def launch_controller(config):
    controller = Controller(scene=config["scene_name"],
                            agentMode=config.get("agent_mode", "bot"),
                            width=config.get("width", 300),
                            height=config.get("height", 300),
                            visibilityDistance=config.get("visibility_distance", 1.5),
                            gridSize=config.get("grid_size", 0.25),
                            renderDepthImage=config.get("render_depth", True),
                            renderClassImage=config.get("render_class", True),
                            renderObjectImage=config.get("render_object", True))
    return controller


def thor_get(controller, *keys):
    """Get the true environment state, which is the metadata returned
    by the controller. If you would like a particular state variable's value,
    pass in a sequence of string keys to retrieve that value.
    For example, to get agent pose, you call:

    env.state("agent", "position")"""
    event = controller.step(action="Pass")
    if len(keys) > 0:
        d = event.metadata
        for k in keys:
            d = d[k]
        return d
    else:
        return event.metadata

def thor_agent_pose2d(controller):
    """Returns a tuple (x, y, th), a 2D pose
    """
    position = thor_get(controller, "agent", "position")
    rotation = thor_get(controller, "agent", "rotation")
    return position["x"], position["z"], rotation["y"]

def thor_agent_pose(controller):
    """Returns a tuple (pos, rot),
    pos: dict (x=, y=, z=)
    rot: dict (x=, y=, z=)
    """
    position = thor_get(controller, "agent", "position")
    rotation = thor_get(controller, "agent", "rotation")
    return position, rotation

def thor_apply_pose(controller, pose):
    """Given a 2d pose (x,y,th), teleport the agent to that pose"""
    pos, rot = thor_agent_pose(controller)
    x, z, th = pose
    # if th != 0.0:
    #     import pdb; pdb.set_trace()
    controller.step("TeleportFull",
                    x=x, y=pos["y"], z=z,
                    rotation=dict(y=th))
    controller.step(action="Pass")  #https://github.com/allenai/ai2thor/issues/538

def thor_object_poses(controller, object_type):
    """Returns a dictionary id->pose
    for the objects of given type. An object pose is
    a 3D (x,y,z) tuple"""
    thor_objects = thor_get(controller, "objects")
    objposes = {}
    for obj in thor_objects:
        if obj["objectType"] == object_type:
            pose = (obj["position"]["x"], obj["position"]["y"], obj["position"]["z"])
            objposes[obj["objectId"]] = pose
    return objposes

def thor_visible_objects(controller):
    thor_objects = thor_get(controller, "objects")
    for obj in thor_objects:
        if obj["visible"]:
            result.append(obj)
    return result

def convert_scene_to_grid_map(controller, scene_name, grid_size):
    """Converts an Ai2Thor scene to a GridMap"""
    x, z = reachable_thor_loc2d(controller)

    # obtain grid indices for coordinates  (origin NOT at (0,0))
    thor_gx = (x // grid_size).astype(int)
    thor_gy = (z // grid_size).astype(int)
    width = max(thor_gx) - min(thor_gx)
    length = max(thor_gy) - min(thor_gy)

    # save these for later use
    thor_gx_range = (min(thor_gx), max(thor_gx))
    thor_gy_range = (min(thor_gy), max(thor_gy))

    # remap coordinates to be nonnegative (origin AT (0,0))
    gx = remap(thor_gx, min(thor_gx), max(thor_gx), 0, width).astype(int)
    gy = remap(thor_gy, min(thor_gy), max(thor_gy), 0, length).astype(int)

    # Little test: can convert back
    try:
        assert all(remap(gx, min(gx), max(gx), thor_gx_range[0], thor_gx_range[1]).astype(int) == thor_gx)
        assert all(remap(gy, min(gy), max(gy), thor_gy_range[0], thor_gy_range[1]).astype(int) == thor_gy)
    except AssertionError as ex:
        print("Unable to remap coordinates")
        raise ex

    # grid map positions
    positions = set(zip(gx, gy))

    # grid map dimensions
    # obstacles: locations that do not fall into valid positions
    obstacles = {(x,y)
                 for x in gx
                 for y in gy
                 if (x,y) not in positions}

    grid_map = GridMap(width, length, obstacles,
                       name=scene_name,
                       ranges_in_thor=(thor_gx_range, thor_gy_range))

    return grid_map


class ThorEnv(pomdp_py.Environment):
    """Maintains Thor scene controller as well as POMDP state.

    Whenever "state" is used as part of a variable, then that
    variable refers to some information in the POMDP world (e.g.
    locations in the grid map)"""
    def __init__(self, robot_id, target_object, thor_config, scene_info, **params):
        """
        robot_id (int) ID for robot
        target_object (tuple) A tuple of (target_id, target_class)
        thor_config (dict) configuration for starting thor controller
        scene_info (dict) see load_scene_info in process_scenes.py
        params (dict) Additional parameters (e.g. rmax, rmin for reward model).
        """
        self.target_object = target_object
        self.robot_id = robot_id

        assert "scene_name" in thor_config, "Scene name required in config."
        self.config = thor_config
        self.scene_info = scene_info
        self.grid_size = thor_config.get("grid_size", 0.25)
        self.controller = launch_controller(thor_config)

        self.grid_map = convert_scene_to_grid_map(self.controller,
                                                  self.config["scene_name"],
                                                  self.grid_size)

        # State in POMDP grid space. Only need to know robot and target state
        # Other states are maintained by the controller (not likely useful to
        # us) If there are multiple objects with the given target class, we
        # maintain all of them and their ids are obtained by incrementing from
        # the given target_id.
        thor_init_robot_pose = thor_agent_pose2d(self.controller)
        init_robot_loc = self.grid_map.to_grid_pos(*thor_init_robot_pose[:2], grid_size=self.grid_size)
        init_robot_pose = (*init_robot_loc, to_rad(thor_init_robot_pose[2]))
        init_robot_state = RobotState(self.robot_id, {"pose": init_robot_pose,
                                                      "loc": init_robot_pose[:2],
                                                      "energy": 0.0,
                                                      "terminal": False})

        target_id, target_type = target_object
        thor_objposes = thor_object_poses(self.controller, target_type)
        if len(thor_objposes) == 0:
            raise ValueError("No object instance for type {}".format(target_type))

        object_states = {self.robot_id: init_robot_state}
        for i, thor_objid in enumerate(thor_objposes):
            objid = target_id + i

            thor_x, thor_y, thor_z = thor_objposes[thor_objid]
            objloc = self.grid_map.to_grid_pos(thor_x, thor_z,
                                               grid_size=self.grid_size)
            objloc = self.grid_map.snap_to_grid(objloc)
            objstate = LocObjState(objid, target_type, {"loc": objloc})
            object_states[objid] = objstate

        init_state = JointState(object_states)

        robot_trans_model = DetRobotTrans(self.robot_id, self.grid_map, schema="vw")
        transition_model = SearchTransitionModel(self.robot_id, robot_trans_model)

        reward_model = ThorSearchRewardModel(self.robot_id, self.target_id, self.grid_map,
                                             rmax=params.get("rmax", 100),
                                             rmin=params.get("rmin", -100))
        super().__init__(init_state, transition_model, reward_model)

    @property
    def target_id(self):
        return self.target_object[0]

    @property
    def target_class(self):
        return self.target_object[1]

    def state_transition(self, action, execute=False):
        """When state transition is executed, the action
        is applied in the real world"""
        next_state = self.transition_model.sample(self.state, action)
        reward = self.reward_model.sample(self.state, action, next_state)
        if execute:
            self.apply_transition(next_state)
            if isinstance(action, Move):
                thor_next_robot_pose = self.grid_map.to_thor_pose(*next_state[self.robot_id].pose,
                                                                  grid_size=self.grid_size)
                thor_apply_pose(self.controller, thor_next_robot_pose)
        return reward

    def provide_observation(self, observation_model, action):
        """Provides an observation based on the THOR simulator.

        Note that an object visible in THOR may not be actually detectable
        in our framework because the agent's detector may have a shorter range
        for that object.

        Args:
            observation_model (MultiDetectorModel)
            action

        Return: JointObz
        """
        # Detectable objects
        detectors = observation_model.detectors
        objects = set().union(*[detectors[did].detectable_objects
                                for did in detectors])

        robot_pose = self.state[self.robot_id].pose

        # For each object type, get the thor pose of its instances. The robot
        # will receive an observation with respect to the closest instance to
        # the robot. The observed location (in POMDP perspective) will be the
        # closest __reachable__ location in the grid map. This is done by
        # constructing an object state with that location.
        object_states = {}
        for objid in objects:
            object_type = self.scene_info.obj_type(objid)
            thor_instance_poses = thor_object_poses(self.controller, object_type)
            instance_poses = set()  # no need for a mapping here because we will just pick one
            for thor_objid in thor_instance_poses:
                thor_x, thor_y, thor_z = thor_instance_poses[thor_objid]
                objloc = self.grid_map.to_grid_pos(thor_x, thor_z,
                                                   grid_size=self.grid_size)
                objloc = self.grid_map.snap_to_grid(objloc)
                instance_poses.add(objloc)
            # Selecting the instance with closest geodesic dist;
            closest_instance_pose = min(instance_poses,
                key=lambda objloc: self.grid_map.geodesic_distance(robot_pose[:2], objloc))
            sobj = LocObjState(objid, object_type, {"loc": closest_instance_pose})
            object_states[objid] = sobj
        env_state = JointState({self.robot_id: self.state[self.robot_id], **object_states})
        return observation_model.sample(env_state, action)



class ThorSearchRewardModel(pomdp_py.RewardModel):
    """Quote from: https://ai2thor.allenai.org/robothor/cvpr-2021-challenge/
    A navigation episode is considered successful if both of the following criteria are met:

        The specified object category is within 1 meter (geodesic distance) from
        the agent's camera, and the agent issues the STOP action, which
        indicates the termination of the episode.  The object is visible from in
        the final action's frame.

    Matt Deike explained: Basically euclidean distance plus checking if wall blocking.

        I will clarify the documentation on this, but we were using Geodesic
        Distance to more or less mean Euclidean Distance without going through a
        wall

    In https://arxiv.org/pdf/1910.14442.pdf:

        The agent has 1000 time steps to achieve this...
    (I will do 200 or 500)

    """
    def __init__(self, robot_id, target_id, grid_map,
                 grid_size=0.25, rmax=100, rmin=-100):
        self.rmax = rmax
        self.rmin = rmin
        self.robot_id = robot_id
        self.target_id = target_id
        self.grid_map = grid_map

        # Number of grids away that the agent can validly declare
        self.declare_dist_grids = 1.0 / grid_size

    def _facing(self, robot_pose, point):
        """Returns true if the robot pose is looking in the direction of point"""
        rx, ry, th = robot_pose
        # point in direction of robot facing
        rx2 = rx + math.sin(th)
        ry2 = ry + math.cos(th)
        px, py = point
        return np.dot(np.array([px - rx, py - ry]),
                      np.array([rx2 - rx, ry2 - ry])) > 0

    def sample(self, state, action, next_state):
        if state[self.robot_id].terminal:
            return 0

        if isinstance(action, Declare):
            robot_loc = state[self.robot_id].loc
            target_loc = state[self.target_id].loc

            # # Closer than threshold
            # if euclidean_dist(robot_loc, target_loc) <= self.declare_dist_grids:
            #     # Not blocked by wall
            #     if not self.grid_map.blocked(robot_loc, target_loc):
            #         # facing the object (not guaranteed to be visible)
            #         if self._facing(state[self.robot_id].pose, target_loc):
            #             return self.rmax
            if action.loc is None:
                decloc = state[self.robot_id].loc
            else:
                decloc = action.loc
            if decloc == state[self.target_id].loc:
                return self.rmax
            else:
                return self.rmin

            # return self.rmin
        else:
            return self.step_reward_func(state, action, next_state)

    def step_reward_func(self, state, action, next_state):
        if next_state[self.robot_id]["energy"] < 0:
            return self.rmin
        else:
            return -1
