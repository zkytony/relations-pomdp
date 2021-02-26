import pomdp_py
from corrsearch.objects import JointObz, ObjectObz
from corrsearch.models.state import *
from corrsearch.models.robot_model import UseDetector
from corrsearch.utils import *

class NullObz(ObjectObz):
    """Didn't observe the object"""
    def __init__(self, objid):
        super().__init__(objid, None, {})

def svar(objid):
    """Convention to name the state variable of an object"""
    return "s" + str(objid)

def id_from_svar(var):
    return int(var[1:])

class DetectorModel(pomdp_py.ObservationModel):
    """Detector Model is the sensor model of the detector.
    The observation should be factored by objects,
    and for each object, we have Pr(zi | si, sr', a).
    This model should have a way to sample and evaluate
    this per-object model

    Convention: An empty observation for an object may be received.  There is no
    notion of "joint null observation". The observation is always factored, and
    could be factored into null observation for every object.
    """
    def __init__(self, detector_id, robot_id, name=None):
        self.id = detector_id
        self.robot_id = robot_id
        if name is None:
            name = "detector_%d" % self.id
        self.name = name

    def detectable(self, objid):
        """Returns true if object `objid` is detectable"""
        raise NotImplementedError

    @property
    def detectable_objects(self):
        raise NotImplementedError

    def iprob(self, objobz, objstate, robot_state, action):
        """
        Returns the probability of Pr(zi | si, sr', a)
        """
        raise NotImplementedError

    def isample(self, objstate, robot_state, action):
        """
        Returns a sample zi according to Pr(zi | si, sr', a)
        """
        raise NotImplementedError

    def probability(self, observation, next_state, action, **kwargs):
        """
        Pr(z | s', a)

        Args:
            observation (JointObz): Joint observation
            next_state (JointState): Joint state
            action (Action): action
        """
        if not isinstance(action, UseDetector)\
           or action.detector_id != self.id:
            # .sample should return the joint of null observations, which is expected
            return indicator(self.sample(next_state, action) == observation)

        p = 1.0
        for objid in observation:
            if objid == self.robot_id:
                continue
            zi = observation[objid]
            si = next_state[objid]
            sr = next_state[self.robot_id]
            p *= self.iprob(zi, si, sr, action)
        return p

    def sample(self, next_state, action, **kwargs):
        """
        z ~ O(s',a)

        Args:
            next_state (JointState): Joint state
            action (Action)
        """
        if not isinstance(action, UseDetector)\
           or action.detector_id != self.id:
            return JointObz({objid:NullObz(objid)
                             for objid in next_state
                             if objid != self.robot_id})

        objzs = {}
        for objid in next_state:
            if objid == self.robot_id:
                continue
            si = next_state[objid]
            sr = next_state[self.robot_id]
            objzs[objid] = self.isample(si, sr, action)
        return JointObz(objzs)


class CorrDetectorModel(pomdp_py.ObservationModel):
    """A detector model for the case where the state space does not
    contain the observed object's state.

    A CorrDetectorModel is specified by a target object id, detector model plus
    a distribution where the probability of one object's state conditioned on
    the target state can be computed. The id of this CorrDetector is the same as
    the given detector model. Also, in order to generate an observation from
    this model, we need access to the list of objects

    More on the distribution: The distribution should be over the object states.
        A state variable is s%d where %d is the object id.  It does not matter,
        from this interface's perspective, what attributes of the object state
        is used for the distribution.

    Note: This is a basic model. It explicitly enumerates
    over the object states. If object states are too large,
    then another more scalable model should be used.

    """
    def __init__(self, target_id, objects, detector_model, dist, compute_conditions=True):
        """
        Args:
            objects (dict): mapping from id (int) to Object
            dist (JointDist): Joint distribution of all objects (including the target)
        """
        self.target_id = target_id
        self.detector_model = detector_model
        self.dist = dist
        self.objects = objects

        # mapping from starget to a distribution Pr(si|starget)
        self.cond_dists = {}
        if compute_conditions:
            # Compute conditional probabilities, for all objects
            # conditioned on every possible value of the target location.x
            target_states = self.dist.valrange(svar(self.target_id))
            for var in self.dist.variables:  # variable is string like 's#'
                if var != svar(self.target_id):
                    for starget in target_states:
                        mg = self.dist.marginal([var],
                                                observation={svar(self.target_id) : starget})
                        self.cond_dists[(var, starget)] = mg.to_tabular_dist()

    @property
    def id(self):
        return self.detector_model.id

    @property
    def robot_id(self):
        return self.detector_model.robot_id

    @property
    def detectable_objects(self):
        return self.detector_model.detectable_objects

    def __repr__(self):
        return "CorrDetectorModel(%d)" % self.detector_model.id

    def dist_si(self, objid, starget):
        """Returns the distribution (JointDist) for Pr(si | starget)"""
        # Compute the Pr(si | starget). TODO: can this be pre-computed?
        if (svar(objid), starget) in self.cond_dists:
            si_dist = self.cond_dists[(svar(objid), starget)]
        else:
            import pdb; pdb.set_trace()
            si_dist = self.dist.marginal([svar(objid)],
                                         observation={svar(self.target_id) : starget})
            # convert to tabular for faster inference
            si_dist = si_dist.to_tabular_dist()
            # cache
            self.cond_dists[(svar(objid), starget)] = si_dist
        return si_dist

    def probability(self, observation, next_state, action, **kwargs):
        """
        Pr(zi | sr, starget, a) = sum_si Pr(zi | sr, si, a) Pr(si | starget)
        """
        if not isinstance(action, UseDetector)\
           or action.detector_id != self.id:
            # .sample should return the joint of null observations, which is expected
            return indicator(self.sample(next_state, action) == observation)

        p = 1.0
        for objid in observation:
            if objid == self.robot_id:
                continue

            zi = observation[objid]
            starget = next_state[self.target_id]
            sr = next_state[self.robot_id]

            if objid in next_state.object_states:
                # Correlation not necessary because object state is already maintained
                p *= self.detector_model.iprob(zi, next_state[objid], sr, action)
                continue

            # sum_si Pr(zi | sr, si, a) * Pr(si | starget)
            dist_si = self.dist_si(objid, starget)
            pi = 0.0
            for si in dist_si.valrange(svar(objid)):
                pi += self.detector_model.iprob(zi, si, sr, action) * dist_si.prob({svar(objid):si})
            p *= pi
        return p

    def sample(self, next_state, action, **kwargs):
        """In order to generate an observation from this model,
        we need access to the list of objects"""
        if not isinstance(action, UseDetector)\
           or action.detector_id != self.id:
            # .sample should return the joint of null observations, which is expected
            return JointObz({objid:NullObz(objid)}
                            for objid in next_state)

        objzs = {}
        for objid in self.objects:
            # First generate object state according to Pr(si | starget),
            # then generate the object observation using the detector model.
            if objid == self.robot_id:
                continue

            sr = next_state[self.robot_id]
            if objid in next_state:
                # objid already contained in state. No need to use correlation.
                objzs[objid] = self.detector_model.isample(next_state[objid], sr, action)
                continue

            starget = next_state[self.target_id]
            dist_si = self.dist_si(objid, starget)
            si = dist_si.sample()[svar(objid)]
            zi = self.detector_model.isample(si, sr, action)
            objzs[objid] = zi
        return JointObz(objzs)

    def get_all_observations(self):
        """The observation space is still the same as the one of the detector model"""
        return self.detector_model.get_all_observations()


class MultiDetectorModel(pomdp_py.ObservationModel):
    def __init__(self, detectors, observations=None):
        """
        Args:
            detectors (list or array-like): a list of detector models (DetectorModel).
            observations (list): List of all possible observations.
        """
        self.observations = observations
        if type(detectors) == list:
            self.detectors = {d.id: d
                              for d in detectors}
        elif type(detectors) == dict:
            self.detectors = detectors
        else:
            raise ValueError("unsupported type for detectors: {}".format(type(detectors)))

    def probability(self, observation, next_state, action, **kwargs):
        if isinstance(action, UseDetector):
            return self.detectors[action.detector_id].probability(observation, next_state, action)
        else:
            all_null = all(isinstance(observation[objid], NullObz)
                           for objid in observation)
            return indicator(all_null)

    def sample(self, next_state, action, **kwargs):
        if isinstance(action, UseDetector):
            return self.detectors[action.detector_id].sample(next_state, action)
        else:
            return JointObz({objid : NullObz(objid)
                             for objid in next_state
                             if not isinstance(next_state[objid], RobotState)})

    def get_all_observations(self):
        obz_set = set()
        obz_list = []
        for d_id in sorted(self.detectors):
            detector = self.detectors[d_id]
            # We assume detector.get_all_observations() always returns observations
            # in the same order every time it is called.
            for z in detector.get_all_observations():
                if z not in obz_set:
                    obz_list.append(z)
                    obz_set.add(z)
        return obz_list
