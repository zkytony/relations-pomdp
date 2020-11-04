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

def compute_detections(observation):
    """Given an observation (CombinedObservation),
    return a set of object classes and ids that are detected.
    A detection results in an ObjectObservation with a 'label'
    that is an integer (the id of the object being detected)."""
    detected_classes = set()
    detected_ids = set()
    for o in observation.observations:
        for objid in o.object_observations:
            objo = o.object_observations[objid]
            if objo["label"] == objo.objclass:
                detected_classes.add(objo.objclass)
                detected_ids.add(objid)
    return detected_classes, detected_ids

class CorrelationObservationModel(pomdp_py.ObservationModel):
    def __init__(self, robot_id):
        self.robot_id = robot_id

    def probability(self, observation, next_state, action,
                    objid=None, class_to_fov={}, grid_map=None, dist_thresh=1):
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

        # detected_classes, detected_ids = compute_detections(observation)

        # for c in detected_classes:
        #     for x, y in class_to_fov[c]:
        pass



    # def sample(self, next_state, action, **kwargs):
    #     pass
