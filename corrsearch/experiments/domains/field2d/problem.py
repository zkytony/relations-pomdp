"""
This is a 2D domain that captures the gist of
what we want for our problem.
"""

import yaml
import pomdp_py
import random
from corrsearch.models import *
from corrsearch.objects import *
from corrsearch.utils import *
from corrsearch.experiments.domains.field2d.detector import *
from corrsearch.experiments.domains.field2d.visualizer import *
from corrsearch.experiments.domains.field2d.belief import *
from corrsearch.experiments.domains.field2d.transition import *

class Field2D(SearchProblem):
    """
    A Field2D problem is defined by:
    - a NxM grid of locations
    - the rest follows SearchProblem
    """
    def __init__(self,
                 dim,
                 objects,
                 joint_dist,
                 robot_id,
                 robot_model,
                 locations=None,
                 target_object=None,
                 name="field2d"):
        """
        Args:
            objects (array-like): List of objects (Object)
            joint_dist (JointDist)
            robot_id (int): Robot id
            robot_model (RobotModel)
        """
        self.dim = dim
        self.name = name
        self.robot_id = robot_id
        self.id2objects = {obj.id : obj for obj in objects}
        if locations is None:
            locations = [(x,y) for x in range(dim[0])
                         for y in range(dim[1])]
        super().__init__(
            locations, objects, joint_dist,
            target_object, robot_model
        )

    def set_target(self, target_id, target_class):
        self.target_object = (target_id, target_class)

    @property
    def target_id(self):
        return self.target_object[0]

    @property
    def target_class(self):
        return self.target_object[1]

    def obj(self, objid):
        return self.id2objects[objid]

    def instantiate(self,
                    init_locs,
                    init_robot_setting,
                    init_belief,
                    **kwargs):
        """
        Args:
            init_locs (dict or string): maps from object id to location,
              Or, "random" if string.
            init_robot_setting (tuple):
               A tuple of two elements: robot_pose and energy
            init_belief (string):
               initial belief over the target location.
               Either completely specified or a string
               that is in {"uniform", "informed", "prior"}.
               If "uniform", then the initial belief state
               will be uniform over all locations. If "informed",
               then it will be completely at the true target location.
               If 'prior", then it follows the distribution in the given joint_dist
        """
        assert self.target_object is not None,\
            "Target object not set."

        # object locations
        if init_locs == "random":
            # Random, according to the distribution
            rnd = random
            if "seed" in kwargs and kwargs["seed"] != None:
                seed = kwargs["seed"]
                rnd = random.Random(seed)

            sample = self.joint_dist.sample(rnd=rnd)
            init_locs = {}
            for obj in self._objects:
                if obj.id != self.robot_id:
                    init_locs[obj.id] = sample[svar(obj.id)]["loc"]

        # init state
        object_states = {}
        for obj in self._objects:
            if obj.id == self.robot_id:
                continue
            loc = init_locs[obj.id]
            si = LocObjState(obj.id, obj["class"],
                             {"loc": loc})
            object_states[obj.id] = si

        init_robot_pose, init_energy = init_robot_setting
        robot_state = RobotState(self.robot_id,
                                 {"loc": init_robot_pose[:2],
                                  "pose": init_robot_pose,
                                  "energy": init_energy,
                                  "terminal": False})
        object_states[self.robot_id] = robot_state
        init_state = JointState(object_states)

        # init belief. Only over the target object
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
                if loc == init_locs[self.target_id]:
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

        if kwargs.get("explicit_enum_states", False):
            target_states = list(belief_hist.keys())
            robot_states = self.robot_model.trans_model.get_all_states()
        else:
            target_states = None
            robot_states = [robot_state]

        transition_model = SearchTransitionModel(
            self.robot_id, self.robot_model.trans_model,
            target_states=target_states)

        robot_belief_hist = {}
        for sr in robot_states:
            robot_belief_hist[sr] = indicator(sr == robot_state)
        init_robot_belief = pomdp_py.Histogram(robot_belief_hist)

        # joint init belief
        init_belief = Field2DBelief(self.robot_id, init_robot_belief,
                                    self.target_id, init_target_belief)


        # reward model
        reward_model = Field2DRewardModel(
            self.robot_id, self.target_id,
            rmax=kwargs.get("rmax", 100),
            rmin=kwargs.get("rmin", -100)
        )

        # observation model. Multiple detectors.
        detectors = []
        for detector_id in self.robot_model.detectors:
            detector = self.robot_model.detectors[detector_id]
            corr_detector = CorrDetectorModel(self.target_id,
                                              {obj.id : obj
                                               for obj in self._objects},
                                              detector,
                                              self.joint_dist)
            detectors.append(corr_detector)
        observation_model = MultiDetectorModel(detectors)

        # policy model. Default is uniform
        policy_model = BasicPolicyModel(self.robot_model.actions,
                                        self.robot_model.trans_model)

        env = pomdp_py.Environment(init_state,
                                   transition_model,
                                   reward_model)
        agent = pomdp_py.Agent(init_belief,
                               policy_model,
                               transition_model,
                               observation_model,
                               reward_model)
        return env, agent


    def visualizer(self, **kwargs):
        return Field2DViz(self, **kwargs)



class Field2DRewardModel(SearchRewardModel):
    def step_reward_func(self, state, action, next_state):
        if next_state[self.robot_id]["energy"] < 0:
            return self.rmin
        else:
            return -1
