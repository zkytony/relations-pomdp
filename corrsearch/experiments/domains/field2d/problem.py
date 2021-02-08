"""
This is a 2D domain that captures the gist of
what we want for our problem.
"""

import yaml
import pomdp_py
import random
from corrsearch.models import *
from corrsearch.objects import *
from corrsearch.experiments.domains.field2d.detector import *
from corrsearch.experiments.domains.field2d.visualizer import *

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
        self.dim = dim
        self.name = name
        self.robot_id = robot_id
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
            locs = random.sample(self.locations, len(self.objects))
            init_locs = {self.objects[i].id : locs[i]
                         for i in range(len(self.objects))}

        # init state
        object_states = {}
        for obj in self.objects:
            loc = init_locs[obj.id]
            si = LocObjState(obj.id, obj["class"],
                             {"loc": loc})
            object_states[obj.id] = si

        init_robot_pose, init_energy = init_robot_setting
        robot_state = RobotState(self.robot_id,
                                 {"loc": init_robot_pose[:2],
                                  "pose": init_robot_pose,
                                  "energy": init_energy})
        object_states[obj.id] = robot_state
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
            prior = self.joint_dist.marignal([svar(self.target_id)])
            for loc in self.locations:
                starget = LocObjState(self.target_id,
                                      self.target_class,
                                      {"loc": loc})
                belief_hist[starget] = prior[starget]
        else:
            raise ValueError("Unsupported initial belief type %s" % init_belief)

        init_target_belief = pomdp_py.Histogram(belief_hist)
        init_robot_belief = pomdp_py.Histogram({robot_state: 1.0})
        init_belief = JointBelief({self.target_id:init_target_belief,
                                   self.robot_id:init_robot_belief})

        # transition model
        transition_model = SearchTransitionModel(
            self.robot_id, self.robot_model.trans_model)

        # reward model
        reward_model = Field2DRewardModel(
            self.robot_id, self.target_id,
            rmax=kwargs.get("rmax", 100),
            rmin=kwargs.get("rmin", -100)
        )

        # observation model. Multiple detectors.
        detectors = []
        for detector in self.robot_model.detectors:
            corr_detector = CorrDetectorModel(self.target_id,
                                              {obj.id : obj
                                               for obj in self.objects},
                                              detector,
                                              self.joint_dist)
            detectors.append(corr_detector)
        observation_model = MultiDetectorModel(detectors)

        # policy model. Default is uniform
        policy_model = pomdp_py.UniformPolicyModel(self.robot_model.actions)

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
        if next_state[self.robot_id]["energy"] <= 0:
            return -self.rmin
        else:
            return 0
