# Observation model; An agent may carry one sensor but underneath there could be
# multiple observation models (because the accuracy of detecting individual
# objects differs)
import pomdp_py
import random
from relpomdp.oopomdp.framework import NullObservation, OOObservation, Condition, OEffect, Objobs

# Observation condition / effects
class CanObserve(Condition):
    """Condition to observe"""
    def satisfy(self, next_state, action):
        return True  # always can
    def __str__(self):
        return "CanObserve"
    def __repr__(self):
        return str(self)


class ObserveEffect(OEffect):
    """On-board sensor observation effect.
    It essentially models Pr(o_i | s_i, s_r, a),
    where o_i is the observation about object i (either i's class, or "free").
    So this model is specified by the true positive and false positive rates
    of object i. It assumes that as soon as the object state s_i is within the field
    of view of the sensor that can detect object i, it will be detected.
    """

    def __init__(self, robot_id, sensor, grid_map, noise_params, sensor_cache=None):
        """
        noise_params (dict): Maps from object class to (alpha, beta) which
            defines the noise level of detecting an object of this class
        gamma (float): The (unnormalized) sensor probability when given object state
            is not in the sensor's field of view. Default is 1.0. Note that
            a functional sensor must satisfy beta < gamma < alpha.
        """
        self.robot_id = robot_id
        self.sensor = sensor
        self.grid_map = grid_map  # should be partial for agent
        self.noise_params = noise_params
        self._sensor_cache = None
        if sensor_cache is not None:
            assert sensor_cache.sensor_name == sensor.name,\
                "SensorCache's name must be equal to the given sensor"
            # Here we enforce sensor cache to serve the given grid map upon creation.
            assert sensor_cache.serving(self.grid_map),\
                "SensorCache must be serving the given grid map."
            self._sensor_cache = sensor_cache

        # Effect name is based on sensor name
        self._name = "ObserveEffect-%s" % sensor.name

    @staticmethod
    def sensor_functioning(alpha, beta):
        return random.uniform(0,1) < alpha / (alpha + beta)

    def probability(self, observation, next_state, action, *args, **kwargs):
        """
        This model considers the observation defined as a set of detections
        for every location in the map; Each detection consists of object class
        and pose. We assume the object is contained in one location. Therefore,
        if a positive detection is received (i.e. label!=free and pose!=None),
        we are also simultaneously receiving (unmodeled) observations at other
        locations with label=free and pose=None. If no positive detection
        occurs within the FOV, then we only know that locations in the FOV
        received a detection of label=free/pose=None, but we don't know the
        detection for locations outside of the FOV, thus treating those as uniform.
        (This is essentially the same observation model as the 3D object search
        paper, except for the part about ``we are also simultaneously...'',
        which is an optional thing that's efficient to do in 2D.
        """
        robot_state = next_state.object_states[self.robot_id]
        prob = 1.0
        for objid in observation.object_observations:
            objo = observation.object_observations[objid]

            if objo.objclass in self.noise_params\
               and objid in next_state.object_states:
                objstate = next_state.object_states[objid]
                true_pos_rate, false_pos_rate = self.noise_params[objo.objclass]

                within_range = self.sensor.within_range(robot_state["pose"], objstate["pose"],
                                                        grid_map=self.grid_map,
                                                        cache=self._sensor_cache)

                if within_range:
                    if objo["pose"] is not None:
                        if objo["pose"] == objstate["pose"]:
                            val = true_pos_rate
                        else:
                            val = 1. - true_pos_rate
                    else:
                        # Didn't detect
                        return 1. - true_pos_rate
                else:
                    # We do not have false positive
                    if objo["pose"] is not None:
                        val = 1. - true_pos_rate
                    else:
                        # No change in belief when object state is not within FOV and
                        # you didn't detect the object
                        val = 1.
                prob *= val
        return prob

    def random(self, next_state, action, byproduct=None):
        """
        Randomly sample an observation, according to the probability defined above.x
        """
        robot_state = next_state.object_states[self.robot_id]

        # We will not model occlusion by objects; only occlusion by walls (which is
        # considered by the sensor itself)
        noisy_obs = {}
        for objid in next_state.object_states:
            objstate = next_state.object_states[objid]
            if objstate.objclass in self.noise_params:
                # We will only observe objects that we have noise parameters for
                true_pos_rate, false_pos_rate = self.noise_params[objstate.objclass]
                label = None
                pose = None
                if self.sensor.within_range(robot_state["pose"], objstate["pose"],
                                            grid_map=self.grid_map,
                                            cache=self._sensor_cache):
                    # observable.
                    if ObserveEffect.sensor_functioning(true_pos_rate, 1. - true_pos_rate):
                        # Sensor functioning.
                        label = objstate.objclass
                        pose = objstate["pose"]
                    else:
                        # Sensor malfunction; not observing it
                        label = "free"
                else:
                    # Will not simulate false positive due to complication;
                    label = "free"
                noisy_obs[objid] = Objobs(objstate.objclass, label=label, pose=pose)
        return OOObservation(noisy_obs)

    @property
    def name(self):
        return self._name

    def __str__(self):
        return "ObserveEffect(%s | %s)"\
            % (self.sensor.name, str(list(self.noise_params.keys())))

    def __repr__(self):
        return str(self)
