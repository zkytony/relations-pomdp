# Observation model; An agent may carry one sensor but underneath there could be
# multiple observation models (because the accuracy of detecting individual
# objects differs)
import pomdp_py
import random
from relpomdp.oopomdp.framework import NullObservation, OOObservation


class ObserveEffect(oopomdp.OEffect):
    def __init__(self, robot_id, sensor, noise_params):
        """
        noise_params (dict): Maps from object class to (alpha, beta) which
            defines the noise level of detecting an object of this class
        """
        self.robot_id = robot_id
        self.sensor = sensor
        self.noise_params = noise_params

    @staticmethod
    def sensor_functioning(alpha, beta):
        return random.uniform(0,1) < alpha / (alpha + beta)

    def random(self, next_state, action, **kwargs):
        robot_state = next_state.object_states[self.robot_id]

        # We will not model occlusion by objects; only occlusion by walls (which is
        # considered by the sensor itself)
        noisy_obs = {}
        for objid in next_state.object_states:
            objstate = next_state.object_states[objid]
            if objstate.objclass in self.noise_params:
                # We will only observe objects that we have noise parameters for
                alpha, beta = self.noise_params[objstate.objclass]
                if sensor.within_range(robot_state["pose"][:2], objstate["pose"]):
                    # observable;
                    objo = ObjectObservation(objstate.objclass, pose=objstate["pose"])
                    if ObserveEffect.sensor_functioning(alpha, beta):
                        # Sensor functioning;
                        objo["label"] = objid
                    else:
                        # Sensor malfunction; not observing it
                        objo["label"] = "free"
                else:
                    objo["label"] = "unknown"
            else:
                noisy_obs[objid] = NullObservation()

        return OOObservation(noisy_obs)

    def probability(self, observation, next_state, action, **kwargs):
        prob = 1.0
        for objid in observation.object_observations:
            objo = observation.object_observations[objid]
            if objo.objclass in self.noise_params:
                alpha, beta = self.noise_params[objo.objclass]
                if objo["label"] == "free":
                    val = beta
                elif objo["label"] == "unknown":
                    val = 1.0
                else:
                    val = alpha
            else:
                if isinstance(objo, NullObservation):
                    val = 1.0
                else:
                    val = 1e-9

            prob *= val
        return prob
