
import yaml
import pomdp_py
import random
import copy
from corrsearch.models import *
from corrsearch.objects import *
from corrsearch.utils import *
from corrsearch.experiments.domains.thor.thor import *
from corrsearch.experiments.domains.thor.detector import *
from corrsearch.experiments.domains.thor.belief import *
from corrsearch.experiments.domains.thor.visualizer import *

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

    def __init__(self, robot_id,
                 target_object,
                 scene_name,
                 scene_info,
                 detectors=None,
                 detectors_spec_path=None,
                 move_actions=MOVE_ACTIONS,
                 grid_size=0.25):
        """
        Note: joint_dist should be grounded to the given scene already.

        Args:
            scene_name (str): Name of scene
            scene_info (dict): Info of the scene. Currently,
                a mapping {object_type -> {objid (pomdp) -> obj_dict}}
            detectors_spec_path (str or dict):
                Path to the .yaml file that specifies the detectors,
                or a dictionary spec.
        """
        self.robot_id = robot_id
        self.target_object = target_object
        assert scene_name == scene_info.scene_name,\
            "Scene name {} does not match what is in scene_info: {}"\
            .format(scene_name, scene_info["scene_name"])
        self.scene_name = scene_name

        config = {
            "scene_name": scene_name,
            "width": 400,
            "height": 400,
            "grid_size": grid_size
        }
        self.env = ThorEnv(robot_id, target_object, config, scene_info)

        # Build detectors, actions, robot transition model, robot model
        if detectors is None and detectors_spec_path is None:
            raise ValueError("Either `detectors` or `detectors_spec_path` must be specified.")
        if detectors is None:
            detectors = parse_detector(scene_info, detectors_spec_path, self.robot_id)
        actions = set(move_actions) | {Declare()}
        actions |= set(UseDetector(detector.id,
                                   name=detector.name,
                                   energy_cost=detector.energy_cost)
                       for detector in detectors)
        robot_trans_model = self.env.transition_model.robot_trans_model
        robot_model = RobotModel(detectors, actions, robot_trans_model)

        # Obtain objects: List of object ids in the detector sensors;
        # Also, given the sensors access to the grid map.
        objects = set()
        for detector in detectors:
            for objid in detector.detectable_objects:
                objects.add(objid)
                detector.sensors[objid].grid_map = self.grid_map

        # Locations where object can be.
        # (We will snap object locations to reachable locations; This still works
        #  in the THOR task because task success depends on distance from the robot
        #  to the object, and it will be valid if the robot makes the declaration
        #  at a close enough reachable location)
        locations = self.env.grid_map.free_locations

        joint_dist = None
        super().__init__(
            locations, objects, joint_dist,
            target_object, robot_model
        )

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
                                              self.joint_dist,
                                              compute_conditions=False)
            detectors.append(corr_detector)
        observation_model = MultiDetectorModel(detectors)

        # policy model. Default is uniform
        policy_model = BasicPolicyModel(self.robot_model.actions,
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
        return {}

    @property
    def grid_map(self):
        return self.env.grid_map
