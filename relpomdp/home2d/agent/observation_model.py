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
    def __init__(self, robot_id, sensor, grid_map, noise_params, gamma=1.0):
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
        self.gamma = gamma

        # Effect name is based on sensor name
        self._name = "ObserveEffect-%s" % sensor.name

    @staticmethod
    def sensor_functioning(alpha, beta):
        return random.uniform(0,1) < alpha / (alpha + beta)

    def random(self, next_state, action, byproduct=None):
        robot_state = next_state.object_states[self.robot_id]

        # We will not model occlusion by objects; only occlusion by walls (which is
        # considered by the sensor itself)
        noisy_obs = {}
        for objid in next_state.object_states:
            objstate = next_state.object_states[objid]
            if objstate.objclass in self.noise_params:
                # We will only observe objects that we have noise parameters for
                alpha, beta = self.noise_params[objstate.objclass]
                objo = Objobs(objstate.objclass, label="unknown")
                if self.sensor.within_range(robot_state["pose"], objstate["pose"],
                                            grid_map=self.grid_map):
                    # observable;
                    if ObserveEffect.sensor_functioning(alpha, beta):
                        # Sensor functioning;
                        objo["label"] = objid
                    else:
                        # Sensor malfunction; not observing it
                        objo["label"] = "free"
                noisy_obs[objid] = objo

        return OOObservation(noisy_obs)

    def probability(self, observation, next_state, action, *args, **kwargs):
        robot_state = next_state.object_states[self.robot_id]
        prob = 1.0
        for objid in observation.object_observations:
            objo = observation.object_observations[objid]
            if objo.objclass in self.noise_params\
               and objid in next_state.object_states:
                objstate = next_state.object_states[objid]
                alpha, beta = self.noise_params[objo.objclass]
                if not (beta < self.gamma < alpha):
                    print("Warning: Parameter setting for sensor seems problematic."\
                          "Expected beta < gamma < alpha.")
                # import pdb; pdb.set_trace()
                if self.sensor.within_range(robot_state["pose"], objstate["pose"],
                                            grid_map=self.grid_map):
                    if objo["label"] == objid:
                        val = alpha
                    else:
                        val = beta
                else:
                    val = self.gamma # uniform
                prob *= val
        return prob

    @property
    def name(self):
        return self._name

    def __str__(self):
        return "ObserveEffect(%s | %s)"\
            % (self.sensor.name, str(list(self.noise_params.keys())))

    def __repr__(self):
        return str(self)
