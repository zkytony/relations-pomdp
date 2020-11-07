# Correlation observation model.
# Basically, it models
#
#  Pr (o_j | s_i, s_r, a)
#
# where o_j could be object j's class, or "free". It
# associates object i's state with the observation of
# object j.
#
# Note that o_j does not contain the location of object j.
# But, the robot state s_r, together with the sensor that
# can detect j, should give you a region where object j
# could be at. Then, every location in this region is a
# possible location of j, which can be used to determine
# how spatially correlated it is.
#
# This observation model is intended for belief update only.
# (It would be interesting, not sure if doable, to use this
# for planning as well).

import pomdp_py
from relpomdp.home2d.planning.test_utils import difficulty, correlation
from relpomdp.home2d.learning.subgoal_generator import correlation_score

def compute_detections(observation, return_poses=False):
    """Given an observation (CombinedObservation),
    return a set of object classes and ids that are detected.
    A detection results in an ObjectObservation with a 'label'
    that is class name.

    `return_poses` set to True if returning also a mapping from
    class name to pose"""
    detected_classes = set()
    detected_ids = set()
    detected_poses = {}
    for o in observation.observations:
        for objid in o.object_observations:
            objo = o.object_observations[objid]
            if objo["label"] == objo.objclass:
                detected_classes.add(objo.objclass)
                detected_ids.add(objid)
                detected_poses[objo.objclass] = objo["pose"]
    if return_poses:
        return detected_classes, detected_ids, detected_poses
    else:
        return detected_classes, detected_ids

class CorrelationObservationModel(pomdp_py.ObservationModel):
    def __init__(self, robot_id, room_types, df_corr):
        """
        robot_id (int): Id of robot
        room_types (set): set of room type names (e.g. Kitchen, Bathroom, etc.)
        df_corr (pd.DataFrame): Stores the correlation scores between classes
        """
        self.robot_id = robot_id
        self.room_types = room_types
        self.df_corr = df_corr

    def _spatially_correlated(self,
                              object_pose, object_class,
                              reference_pose, reference_class,
                              grid_map):
        """
        Returns True if the object (e.g. salt) and the reference (e.g. pepper, or Kitchen)
        are spatially correlated, on given grid_map
        """
        # Actually it seems like I will just check the room
        # Not sure why passing in the class names. But maybe useful later?
        if object_class in self.room_types\
           and reference_class in self.room_types:
            return False
        return grid_map.same_room(object_pose, reference_pose)


    def probability(self, observation, next_state, action,
                    objid=None, grid_map=None):
        """
        Args:
            observation (CombinedObservation): An compound observation
                that stores the result of all detections.
            next_state (OOState): Stores the robot state and the state of
                an object (with id given by `objid`)
            objid (int): Object id (required)
            class_to_fov (dict): Maps from object class name to a set of grids
                inside the field of view of the sensor that detects it.
        """
        # For each detected class, iterate over the field of view of the sensor
        # that can detect this class. For each field-of-view location, compute
        # the binary variable of 'near' (using the same function in compute_correlations.py).
        # If 'near' is true, and correlation between the detected class and the
        # given object class is high, then return a high probability. Otherwise,
        # return a low probability.
        given_object_state = next_state.object_states[objid]
        given_class_pose = given_object_state["pose"]

        detected_classes, detected_ids, detected_poses = compute_detections(observation,
                                                                            return_poses=True)
        prob = 1.0
        for detected_class in detected_classes:
            detected_class_pose = detected_poses[detected_class]
            if detected_class_pose == "IN_FOV":
                # TODO: FIX THIS
                detected_class_pose = next_state.object_states[self.robot_id]["pose"][:2]

            correlated = self._spatially_correlated(given_class_pose, given_object_state.objclass,
                                                    detected_class_pose, detected_class,
                                                    grid_map)
            score = correlation_score(given_object_state.objclass,
                                      detected_class, self.df_corr)
            if score > 0.5:
                # the two classes are correlated in the training environments
                if correlated:
                    # the poses of these classes here are still (spatially) correlated
                    val = 1.0
                else:
                    # the poses of these classes are not spatially correlated
                    val = 1e-9
            else:
                # the two classes are not correlated in the training environments
                if correlated:
                    # the poses of these classes here are (spatially) correlated
                    val = 1e-9
                else:
                    # the poses of these classes are still not spatially correlated
                    val = 1.0
            prob *= val
        return prob
