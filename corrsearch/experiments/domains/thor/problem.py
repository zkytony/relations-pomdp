
import yaml
import pomdp_py
import pickle
import random
import copy
from corrsearch.models import *
from corrsearch.objects import *
from corrsearch.utils import *
from corrsearch.experiments.domains.thor.thor import *
from corrsearch.experiments.domains.thor.detector import *
from corrsearch.experiments.domains.thor.belief import *
from corrsearch.experiments.domains.thor.visualizer import *
from corrsearch.experiments.domains.thor.parser import *
from corrsearch.experiments.domains.thor.topo_maps import TopoMap
from corrsearch.experiments.domains.thor.transition import TopoPolicyModel

MOVE_ACTIONS=set(
    [Move((1.0, 0.0), "forward"),
     Move((-1.0, 0.0), "backward"),
     Move((0.0, -math.pi/4), "left"),
     Move((0.0, math.pi/4), "right")]
)

class ThorSearch(SearchProblem):
    """Different from Field2D, the specification of domain highly depends on
    starting the controller of the scene so that information can be obtained.
    So for this domain, the environment is built when the problem is initialized.

    Quote from: https://ai2thor.allenai.org/robothor/cvpr-2021-challenge/
    A navigation episode is considered successful if both of the following criteria are met:

        The specified object category is within 1 meter (geodesic distance) from
        the agent's camera, and the agent issues the STOP action, which
        indicates the termination of the episode.  The object is visible from in
        the final action's frame.
    """

    def __init__(self,
                 robot_id,
                 target_object,
                 scene_info,
                 env,
                 locations,
                 objects,
                 joint_dist,
                 robot_model,
                 grid_size):

        self.robot_id     = robot_id
        self.scene_info   = scene_info
        self.env          = env
        self.grid_size    = grid_size
        super().__init__(
            locations, objects, joint_dist,
            target_object, robot_model
        )

    @classmethod
    def parse(cls, spec_or_path, scene_data_path="data", topo_dir_path="data/topo"):
        """
        Create a ThorSearch problem given spec (or path to .yaml file for spec)
        """
        if isinstance(spec_or_path, str):
            with open(spec_or_path) as f:
                spec = yaml.load(f, Loader=yaml.Loader)
        else:
            spec = spec_or_path

        robot_id = spec["robot_id"]
        scene_name = spec["scene_name"]
        scene_info = load_scene_info(scene_name, data_path=scene_data_path)
        target_class = spec["target_class"]
        target_id = scene_info.objid_for_type(target_class)
        target_object = (target_id, target_class)
        grid_size = spec["grid_size"]
        boundary_thickness = spec["boundary_thickness"]

        detectors = parse_detectors(scene_info, spec["detectors"], robot_id)
        detect_actions = set(UseDetector(detector.id,
                                         name=detector.name,
                                         energy_cost=detector.energy_cost)
                             for detector in detectors)
        actions = {"detect_actions": detect_actions,
                   "declare_actions": {Declare()}}

        # load topological map, if specified
        topo_map = None
        if spec["move_schema"] == "topo":
            topo_file = os.path.join(topo_dir_path, "{}-topo.json".format(scene_name))
            topo_map = TopoMap.load(topo_file)
            actions["rotate_actions"] = parse_move_actions(spec["rotate_actions"])
        else:
            actions["move_actions"] = parse_move_actions(spec["move_actions"])

        config = {
            "scene_name": scene_name,
            "width": 400,
            "height": 400,
            "grid_size": grid_size,
            "agent_mode": "default"
        }
        env = ThorEnv(robot_id, target_object, config, scene_info, topo_map=topo_map)

        robot_trans_model = env.transition_model.robot_trans_model
        robot_trans_model.schema = spec["move_schema"]
        robot_model = RobotModel(detectors, actions, robot_trans_model)

        # Obtain objects: List of object ids in the detector sensors;
        # Also, given the sensors access to the grid map.
        objects = {robot_id: Object(robot_id,
                                    {"class":"robot", "color": [30, 30, 200]})}
        for detector in detectors:
            for objid in detector.detectable_objects:
                objects[objid] = scene_info.obj(objid)
                detector.sensors[objid].grid_map = env.grid_map

        # Locations where object can be.
        locations = copy.deepcopy(env.grid_map.free_locations)
        locations |= env.grid_map.boundary_cells(thickness=boundary_thickness)

        if spec.get("joint_dist_path", None) is not None\
           and os.path.exists(spec["joint_dist_path"]):
            print("Joint Distribution exists. Loading")
            with open(spec["joint_dist_path"], "rb") as f:
                joint_dist = pickle.load(f)
        else:
            print("Creating Joint Distribution...")
            thor_locations = set(env.grid_map.to_thor_pos(*loc, grid_size=grid_size)
                                 for loc in locations)
            joint_dist = parse_dist(scene_info, env.grid_map, thor_locations, spec["probability"],
                                    grid_size=grid_size)
        return ThorSearch(robot_id, target_object, scene_info, env,
                          locations, objects, joint_dist, robot_model, grid_size)

    @property
    def target_id(self):
        return self.target_object[0]

    @property
    def target_class(self):
        return self.target_object[1]


    def instantiate(self, init_belief, **kwargs):
        """
        Instantiate an instance of the search problem in a Thor scene.
        """
        # The main job here is to build an agent

        belief_hist = {}
        if init_belief == "uniform":
            for loc in self.locations:
                starget = LocObjState(self.target_id,
                                      self.target_class,
                                      {"loc": loc})
                belief_hist[starget] = 1 / len(self.locations)
        elif init_belief == "informed":
            epsilon = kwargs.get("epsilon", 1e-12)
            for loc in self.locations:
                starget = LocObjState(self.target_id,
                                      self.target_class,
                                      {"loc": loc})
                if loc == self.env.state[self.target_id].loc:
                    belief_hist[starget] = 1 - epsilon
                else:
                    belief_hist[starget] = epsilon
        elif init_belief == "prior":
            prior = self.joint_dist.marginal([svar(self.target_id)])
            for loc in self.locations:
                starget = LocObjState(self.target_id,
                                      self.target_class,
                                      {"loc": loc})
                belief_hist[starget] = prior.prob({svar(self.target_id):starget})
        else:
            raise ValueError("Unsupported initial belief type %s" % init_belief)

        init_target_belief = pomdp_py.Histogram(belief_hist)
        transition_model = SearchTransitionModel(
            self.robot_id, self.robot_model.trans_model)

        robot_state = copy.deepcopy(self.env.state[self.robot_id])
        init_robot_belief = pomdp_py.Histogram({robot_state:1.0})
        init_belief = ThorBelief(self.robot_id, init_robot_belief,
                                 self.target_id, init_target_belief)
        reward_model = copy.deepcopy(self.env.reward_model)

        # observation model. Multiple detectors.
        detectors = []
        for detector_id in self.robot_model.detectors:
            detector = self.robot_model.detectors[detector_id]
            corr_detector = CorrDetectorModel(self.target_id,
                                              self._objects,
                                              detector,
                                              self.joint_dist)
            detectors.append(corr_detector)
        observation_model = MultiDetectorModel(detectors)

        # policy model. Default is uniform
        if self.robot_model.trans_model.schema == "topo":
            topo_map = self.robot_model.trans_model.topo_map
            policy_model = TopoPolicyModel(self.robot_id, topo_map, self.grid_map,
                                           self.robot_model.actions["rotate_actions"],
                                           self.robot_model.actions["detect_actions"],
                                           self.robot_model.actions["declare_actions"],
                                           grid_size=self.grid_size)
        else:
            actions = self.robot_model.actions["move_actions"]\
                      | self.robot_model.actions["detect_actions"]\
                      | self.robot_model.actions["declare_actions"]
            policy_model = BasicPolicyModel(actions,
                                            self.robot_model.trans_model)

        agent = pomdp_py.Agent(init_belief,
                               policy_model,
                               transition_model,
                               observation_model,
                               reward_model)

        return self.env, agent

    def visualizer(self, **kwargs):
        return ThorViz(self, **kwargs)

    def obj(self, objid):
        return self._objects[objid]

    @property
    def grid_map(self):
        return self.env.grid_map
