
import yaml
import pomdp_py
import random
import copy
from corrsearch.models import *
from corrsearch.objects import *
from corrsearch.utils import *
from corrsearch.experiments.domains.thor.thor import *
from corrsearch.experiments.domains.thor.belief import *

# def build_robot_model():
#     pass

# def parse_joint_dist():
#     pass

# def parse_detectors(detectors_spec):
#     detectors = []
#     for i in range(len(spec["detectors"])):
#         # Build a RangeDetector
#         dspec = spec["detectors"][i]
#         sensors = {}
#         for ref in dspec["sensors"]:
#             # ref can be either an object id or a classname.
#             # If classname, then all objects with that class will
#             # be paired with this sensor.
#             if type(ref) == int:
#                 objids = [ref]
#             else:
#                 # ref is taken as object class
#                 objids = idbyclass[ref]
#             for objid in objids:
#                 sensor_spec = dspec["sensors"][ref]
#                 sensors[objid] = parse_sensor(sensor_spec)

#         params = {}
#         for param_name in dspec["params"]:
#             pspec = dspec["params"][param_name]
#             params[param_name] = {}
#             if type(pspec) == dict:
#                 for ref in pspec:
#                     if type(ref) == int:
#                         objids = [ref]
#                     else:
#                         objids = idbyclass[ref]
#                     for objid in objids:
#                         if objid not in pspec:
#                             print("Warning: Parameter %s unspecified for object %d"\
#                                   % (param_name, objid))
#                             continue
#                         params[param_name][objid] = pspec[ref]
#         detectors.append(RangeDetector(dspec["id"], robot_id,
#                                        dspec["type"], sensors,
#                                        energy_cost=dspec.get("energy_cost", 0),
#                                        name=dspec["name"],
#                                        locations=locations,
#                                        objects=objects,
#                                        **params))


# def parse_sensor(sensor_spec):
#     """Build sensor given sensor_space (dict)"""
#     if sensor_spec["type"] == "fan":
#         sensor = FanSensor(**sensor_spec["params"])
#     elif sensor_spec["type"] == "disk":
#         sensor = DiskSensor(**sensor_spec["params"])
#     else:
#         raise ValueError("Unrecognized sensor type %s" % sensor_spec["type"])
#     return sensor


class ThorSearch(SearchProblem):
    """
    Different from Field2D, the specification of domain highly depends on
    starting the controller of the scene so that information can be obtained.
    So for this domain, the environment is built when the problem is initialized.
    """

    def __init__(self, robot_id,
                 target_object,
                 scene_name,
                 objects,
                 actions,
                 # joint_dist_spec=None,
                 # joint_dist_path=None,
                 # detectors_spec=None,
                 # detectors=None,
                 grid_size=0.25):
        """
        Note: joint_dist should be grounded to the given scene already.

        Args:
            objects (array-like): List of objects (Object)
        """
        self.robot_id = robot_id
        self.target_object = target_object
        self.id2objects = {obj.id : obj for obj in objects}
        self.scene_name = scene_name

        config = {
            "scene_name": scene_name,
            "width": 400,
            "height": 400,
            "grid_size": grid_size
        }
        self.env = ThorEnv(robot_id, target_object, config)

        # # Locations where object can be.
        boundary = self.env.grid_map.boundary_cells(thickness=1)
        locations = self.env.grid_map.free_locations | boundary

        joint_dist = None
        robot_model = None#self.env.transition_model.robot_trans_model
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
        robot_trans_model = DetRobotTrans(self.robot_id, self.env.grid_map, schema="vw")
        transition_model = SearchTransitionModel(
            self.robot_id, robot_trans_model)

        robot_state = copy.deepcopy(self.env.state[self.robot_id])
        init_robot_belief = pomdp_py.Histogram({robot_state:1.0})
        init_belief = ThorBelief(self.robot_id, init_robot_belief,
                                 self.target_id, init_target_belief)
        reward_model = copy.deepcopy(self.env.reward_model)

        # # observation model. Multiple detectors.
        # detectors = []
        # for detector_id in self.robot_model.detectors:
        #     detector = self.robot_model.detectors[detector_id]
        #     corr_detector = CorrDetectorModel(self.target_id,
        #                                       {obj.id : obj
        #                                        for obj in self._objects},
        #                                       detector,
        #                                       self.joint_dist)
        #     detectors.append(corr_detector)
        # observation_model = MultiDetectorModel(detectors)

        print("YO")


    def obj(self, objid):
        return {}
