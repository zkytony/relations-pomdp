"""
Functions related to THOR simulation
"""
import pomdp_py
import numpy as np
import math
import os
import pickle
import yaml
from ai2thor.controller import Controller
from corrsearch.experiments.domains.thor.grid_map import GridMap
from corrsearch.experiments.domains.thor.transition import DetRobotTrans, TopoRobotTrans
from corrsearch.utils import remap, to_rad, euclidean_dist
from corrsearch.models.transition import *
from corrsearch.models.robot_model import *
from corrsearch.models.detector import *
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
                            width=800,#config.get("width", 300),
                            height=800,#config.get("height", 300),
                            visibilityDistance=config.get("visibility_distance", 5.0),
                            fieldOfView=config.get("fov", 120),
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
                    rotation=dict(y=th),
                    horizon=0, standing=True)
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
    result = []
    for obj in thor_objects:
        if obj["visible"]:
            result.append(obj)
    return result

def robothor_scene_names(scene_type="Train", levels=None, nums=None):
    scenes = []
    if scene_type == "Train":
        if levels is None:
            levels = range(1, 13)
        if nums is None:
            nums = range(1, 6)
    elif scene_type == "Val":
        if levels is None:
            levels = range(1, 4)
        if nums is None:
            nums = range(1, 6)
    else:
        raise ValueError("RoboThor has no scene type {}".format(scene_type))

    for i in levels:
        for j in nums:
            scene = "FloorPlan_{}{}_{}".format(scene_type, i, j)
            scenes.append(scene)
    return scenes

def ithor_scene_names(scene_type="kitchen", levels=None):
    scenes = dict(
        kitchen = [f"FloorPlan{i}" for i in range(1, 31)],
        living_room = [f"FloorPlan{200 + i}" for i in range(1, 31)],
        bedroom = [f"FloorPlan{300 + i}" for i in range(1, 31)],
        bathroom = [f"FloorPlan{400 + i}" for i in range(1, 31)]
    )
    if scene_type.lower() in scenes:
        if levels is None:
            return scenes[scene_type]
        else:
            return [scenes[scene_type][i-1] for i in levels]
    raise ValueError("Unknown scene type {}".format(scene_type))


def convert_scene_to_grid_map(controller, scene_info, grid_size):
    """Converts an Ai2Thor scene to a GridMap"""
    x, z = reachable_thor_loc2d(controller)

    # obtain grid indices for coordinates  (origin NOT at (0,0))
    thor_gx = np.round(x / grid_size).astype(int)
    thor_gy = np.round(z / grid_size).astype(int)
    width = max(thor_gx) - min(thor_gx) + 1
    length = max(thor_gy) - min(thor_gy) + 1

    # save these for later use
    thor_gx_range = (min(thor_gx), max(thor_gx) + 1)
    thor_gy_range = (min(thor_gy), max(thor_gy) + 1)

    # remap coordinates to be nonnegative (origin AT (0,0))
    gx = remap(thor_gx, thor_gx_range[0], thor_gx_range[1], 0, width).astype(int)
    gy = remap(thor_gy, thor_gy_range[0], thor_gy_range[1], 0, length).astype(int)

    gx_range = (min(gx), max(gx)+1)
    gy_range = (min(gy), max(gy)+1)

    # Little test: can convert back
    try:
        assert all(remap(gx, gx_range[0], gx_range[1], thor_gx_range[0], thor_gx_range[1]).astype(int) == thor_gx)
        assert all(remap(gy, gy_range[0], gy_range[1], thor_gy_range[0], thor_gy_range[1]).astype(int) == thor_gy)
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
                       name=scene_info.scene_name,
                       ranges_in_thor=(thor_gx_range, thor_gy_range))

    return grid_map


VALID_DECLARE_DISTANCE = 1.0
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
        scene_info (dict) see load_scene_info in thor.py
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
                                                  scene_info,
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
            objstate = LocObjState(objid, target_type, {"loc": objloc})
            object_states[objid] = objstate

        init_state = JointState(object_states)

        if params.get("topo_map", None) is not None:
            topo_map = params["topo_map"]
            self.topo_map = topo_map
            robot_trans_model = TopoRobotTrans(self.robot_id, topo_map, self.grid_map,
                                               grid_size=self.grid_size)
        else:
            robot_trans_model = DetRobotTrans(self.robot_id, self.grid_map)
        transition_model = SearchTransitionModel(self.robot_id, robot_trans_model)

        reward_model = ThorSearchRewardModel(self.robot_id, self.target_id, self.grid_map,
                                             rmax=params.get("rmax", 100),
                                             rmin=params.get("rmin", -100),
                                             grid_size=self.grid_size,
                                             valid_declare_distance=VALID_DECLARE_DISTANCE)
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
                instance_poses.add(objloc)
            # Selecting the instance with closest geodesic dist;
            closest_instance_pose = min(instance_poses,
                key=lambda objloc: self.grid_map.geodesic_distance(robot_pose[:2], objloc))
            sobj = LocObjState(objid, object_type, {"loc": closest_instance_pose})
            object_states[objid] = sobj
        env_state = JointState({self.robot_id: self.state[self.robot_id], **object_states})

        # If an object is actually NOT observable in THOR, set the observation to be Null.
        sim_obz = observation_model.sample(env_state, action)
        visible_objids = set(obj["objectId"]
                             for obj in thor_visible_objects(self.controller))
        object_obzs = {}
        for objid in sim_obz:
            if self.scene_info.to_thor_objid(objid) not in visible_objids:
                object_obzs[objid] = NullObz(objid)
            else:
                object_obzs[objid] = sim_obz[objid]
        actual_obz = JointObz(object_obzs)
        return actual_obz


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
                 grid_size=0.25, rmax=100, rmin=-100, valid_declare_distance=1.0):
        self.rmax = rmax
        self.rmin = rmin
        self.robot_id = robot_id
        self.target_id = target_id
        self.grid_map = grid_map

        # Number of grids away that the agent can validly declare
        self.declare_dist_grids = valid_declare_distance / grid_size

    def _facing(self, robot_pose, point):
        """Returns true if the robot pose is looking in the direction of point"""
        rx, ry, th = robot_pose
        if (rx,ry) == point:
            return True
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

            # Closer than threshold
            if euclidean_dist(robot_loc, target_loc) <= self.declare_dist_grids:
                # Not blocked by wall
                if not self.grid_map.blocked(robot_loc, target_loc):
                    # facing the object (not guaranteed to be visible)
                    if self._facing(state[self.robot_id].pose, target_loc):
                        return self.rmax
            return self.rmin
        else:
            return self.step_reward_func(state, action, next_state)

    def step_reward_func(self, state, action, next_state):
        return -1 - action.energy_cost

class ThorSceneInfo:
    def __init__(self, scene_name, type_obj_map):
        self.scene_name = scene_name
        self.type_obj_map = type_obj_map

        # Map from objid (pomdp) to thor object dict
        self._idp2t = {}
        self._idt2p = {}
        for objtype in self.type_obj_map:
            for objid in self.type_obj_map[objtype]:
                assert objid not in self._idp2t
                self._idp2t[objid] = self.type_obj_map[objtype][objid]
                thor_objid = self.type_obj_map[objtype][objid]["objectId"]
                self._idt2p[thor_objid] = objid

    def obj_types(self):
        return set(self.type_obj_map.keys())

    def pomdp_objids(self, objtype):
        """Returns objids (pomdp) for the given object type"""
        return set(self.type_obj_map[objtype].keys())

    def objid_for_type(self, objtype):
        """Returns a pomdp object id for the given type;
        If there are multiple ones, return the smallest."""
        return min(self.pomdp_objids(objtype))

    def obj(self, objid):
        """Returns the Object with thor data structure given objid (pomdp)"""
        return Object(objid, self._idp2t[objid])

    def obj_type(self, objid):
        return self._idp2t[objid]["objectType"]

    @property
    def objects(self):
        return self._idp2t

    def thor_obj_pose2d(self, objid):
        obj = self.obj(objid)
        thor_pose = obj["position"]["x"], obj["position"]["z"]
        return thor_pose

    def to_thor_objid(self, objid):
        return self._idp2t[objid]["objectId"]

    @classmethod
    def load(cls, scene_name, data_path="data"):
        """Returns scene info, which is a mapping:
        A mapping {object_type -> {objid (pomdp) -> obj_dict}}"""
        with open(os.path.join(
                data_path, "{}-objects.pkl".format(scene_name)), "rb") as f:
            type_obj_map = pickle.load(f)
        with open(os.path.join(os.path.dirname(data_path), "config", "colors.yaml")) as f:
            colors = yaml.load(f)
        scene_info = ThorSceneInfo(scene_name, type_obj_map)
        for objid in scene_info.objects:
            obj = scene_info.objects[objid]
            obj["color"] = colors.get(obj["objectType"], [128, 128, 128])
            obj["class"] = obj["objectType"]
        return scene_info

    @classmethod
    def extract_objects_info(cls, scene):
        """Returns a mapping from object type to {objid -> objinfo}
        where `objid` is an integer id and `objinfo` is the metadata
        of the object obtained at the initial state of the scene"""
        controller = launch_controller({"scene_name": scene})
        event = controller.step(action="Pass")

        # Maps from object type to an integer (e.g. 1000) that
        # is the the id of the FIRST INSTANCE of that type
        type_id_map = {}
        # Maps from object type to objects
        type_obj_map = {}
        for obj in event.metadata["objects"]:
            objtype = obj["objectType"]
            if objtype not in type_id_map:
                type_id_map[objtype] = (len(type_id_map)+1)*1000
                type_obj_map[objtype] = {}

            objid = len(type_obj_map[objtype]) + type_id_map[objtype]
            type_obj_map[objtype][objid] = obj

        controller.stop()
        return type_obj_map

    @classmethod
    def shared_objects_in_scenes(cls, scenes, data_path="data"):
        objects = None
        for scene in scenes:
            scene_info = ThorSceneInfo.load(scene, data_path=data_path)
            if objects is None:
                objects = scene_info.obj_types()
            else:
                objects = objects.intersection(scene_info.obj_types())
        return objects

load_scene_info = ThorSceneInfo.load
