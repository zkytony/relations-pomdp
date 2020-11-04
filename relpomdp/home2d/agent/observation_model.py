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

    def __init__(self, robot_id, sensor, grid_map, noise_params):
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

        # Effect name is based on sensor name
        self._name = "ObserveEffect-%s" % sensor.name

    @staticmethod
    def sensor_functioning(alpha, beta):
        return random.uniform(0,1) < alpha / (alpha + beta)

    def probability(self, observation, next_state, action, *args, **kwargs):
        """
        An observation model that considers noise. This model is not exactly
        the same as the one used in MOS-3D.

        The observation for each object is (pose, label), with the following two cases:
        - (pose=None, label="free"): The object is not detected.
        - (pose=(x,y), label=Objclass): The object is detected with pose at (x,y)
        Note that either case could happen because the sensor behaved correctly,
        or that the sensor malfunction. This quality is specified by the true positive
        and false negative rates per class.

        If the observation has pose=(x,y) for object i, then object i's pose must
        be at (x,y) in the given state in `next_state`. Otherwise, the probability is
        zero (actually 1e-9 to avoid numerical issues). This matches the way the pose
        is set during sampling (see random() below).

        There are pros and cons of using this model.
        Pros:
        - We can explicitly model true/false positive and true/false negative parameters.
          In the MOS-3D observation model, we could only consider true positive and false negative.
        Cons:
        - An observation cannot be associated with a grid cell. That means when
          the robot detects an Object, there'll be equal likelihood in every location
          within the FOV for where the object is. And the robot must either (1) rely
          on some other ability to estimate the pose of the object (2) move around to
          have its FOV cover different grid cells, and see if it receives the same observation.
          This may lead to poor decision of taking the 'find' action.
        - Updating the belief using this model requires iterating over the entire state space.

        As a comparison, there are some pros and cons of the model I used before in MOS3D
        Pros:
        - Reduces a set of voxel-label tuples to just one voxel-label tuple, for efficient
          observation sampling and belief update
        - Considers the pose of the voxel, so won't have the cons above.
        - Updating the belief using this model requires only checking the grid cells within the FOV.
        Cons:
        - Cannot model true negative and false positive, because we treat locations outside
          of the FOV to contribute nothing to the belief update (because we don't have the
          information of the d(si) variable, thus the probability over d(si) is uniform.
        - May require over-estimate of the sensor's accuracy. Because a low true positivity rate
          would lead to the robot not believing in the object's existence, even after receiving
          an observation about it; This is because the belief outside of the FOV is not reduced
          as a result of this observation. This leads to a behavior where the robot must look
          at the object multiple times in order to be certain, which is a reasonable behavior,
          but when the true positivity rate is not substantially larger than the uniform probability,
          (e.g. at least 10 to 1), then the robot has trouble believing in its observation, even
          though the true positivity rate is already quite high, and such belief update behavior
          varies dependent on the world size because the true positivity rate must overcome the
          cumulative belief in locations outside of the FOV.

        I don't think either one is right or wrong. It is a matter of balance between pros/cons.
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
                                                        grid_map=self.grid_map)
                if within_range:
                    if objo["label"] == objstate.objclass:

                        val = true_pos_rate
                    else:
                        val = 1. - true_pos_rate
                else:
                    if objo["label"] == "free":
                        val = 1. - false_pos_rate
                    else:
                        val = false_pos_rate
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
                                            grid_map=self.grid_map):
                    # observable.
                    if ObserveEffect.sensor_functioning(true_pos_rate, 1. - true_pos_rate):
                        # Sensor functioning.
                        label = objstate.objclass
                        pose = objstate["pose"]
                    else:
                        # Sensor malfunction; not observing it
                        label = "free"
                else:
                    if ObserveEffect.sensor_functioning(1 - false_pos_rate, false_pos_rate):
                        label = "free"
                    else:
                        label = objstate.objclass
                        # False positive. exact pose is unsure, but in FOV
                        # TODO: You should actually simulated a pose within the FOV.
                        pose = "IN_FOV"#robot_state["pose"][:2]
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
