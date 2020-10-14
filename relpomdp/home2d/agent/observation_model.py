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


class ObserveEffect(OEffect):
    def __init__(self, robot_id, sensor, grid_map, noise_params):
        """
        noise_params (dict): Maps from object class to (alpha, beta) which
            defines the noise level of detecting an object of this class
        """
        self.robot_id = robot_id
        self.sensor = sensor
        self.grid_map = grid_map  # should be partial for agent
        self.noise_params = noise_params

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
                objo = Objobs(objstate.objclass, pose=objstate["pose"])
                if self.sensor.within_range(robot_state["pose"], objstate["pose"],
                                            grid_map=self.grid_map):
                    # observable;
                    if ObserveEffect.sensor_functioning(alpha, beta):
                        # Sensor functioning;
                        objo["label"] = objid
                    else:
                        # Sensor malfunction; not observing it
                        objo["label"] = "free"
                else:
                    objo["label"] = "unknown"
                noisy_obs[objid] = objo
            # else:
            #     noisy_obs[objid] = NullObservation()

        return OOObservation(noisy_obs)

    def probability(self, observation, next_state, action, *args, **kwargs):
        robot_state = next_state.object_states[self.robot_id]
        prob = 1.0
        for objid in observation.object_observations:
            objo = observation.object_observations[objid]
            objstate = next_state.object_states[objid]
            if objo.objclass in self.noise_params:
                alpha, beta = self.noise_params[objo.objclass]
                # import pdb; pdb.set_trace()
                if self.sensor.within_range(robot_state["pose"], objstate["pose"],
                                            grid_map=self.grid_map):
                    # object pose is observable
                    if objo["pose"] == objstate["pose"]:
                        if objo["label"] == objid:
                            val = alpha
                        else:
                            val = beta
                    else:
                        # given that object is at pose in state but
                        # didn't observe it at that location is not
                        # a probable event.
                        val = 1e-9
                else:
                    val = 1.0 # gamma
                prob *= val
        return prob
