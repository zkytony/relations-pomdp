import yaml
import pomdp_py
import itertools
import pickle
from corrsearch.models.problem import *
from corrsearch.models.detector import *
from corrsearch.models.robot_model import *
from corrsearch.probability import TabularDistribution, FactorGraph
from corrsearch.objects.template import Object
from corrsearch.experiments.domains.field2d.detector import *
from corrsearch.experiments.domains.field2d.problem import Field2D
from corrsearch.experiments.domains.field2d.transition import *
from corrsearch.experiments.domains.field2d.spatial_relations import *

def parse_sensor(sensor_spec):
    """Build sensor given sensor_space (dict)"""
    if sensor_spec["type"] == "fan":
        sensor = FanSensor(**sensor_spec["params"])
    elif sensor_spec["type"] == "disk":
        sensor = DiskSensor(**sensor_spec["params"])
    else:
        raise ValueError("Unrecognized sensor type %s" % sensor_spec["type"])
    return sensor

def parse_domain(spec):
    """
    Parse the domain from the spec (dict).
    Care about "objects", "detectors", "name", "bg", "dim".

    In our domain, the agent can have multiple "detectors",
    each corresponds to a DetectorModel that can induce a distribution
    over all object observations. A DetectorModel can have multiple
    underlying SENSORS, each with their geometry, properties, and
    capabilities (i.e. which objects are detectable by the sensor).
    These can all be specified by the spec in the "detectors" entry.
    It should contain a list of detectors, and each detector contains
    a mapping of sensors. See config/simple_config.yaml for an example.

    Args:
        spec (dict). Contains
    Return:
        Field2D problem, so that Field2D can be created and POMDPs can be instantiated.
    """
    name = spec["name"]
    dim = spec["dim"]
    # enumerate list of locations
    locations = [(x,y) for x in range(dim[0]) for y in range(dim[1])]

    objects = []
    robot_id = None
    idbyclass = {}
    id2objects = {}
    for i in range(len(spec["objects"])):
        objspec = spec["objects"][i]
        obj = Object(objspec["id"], objspec)
        objects.append(obj)
        if objspec["class"] == "robot":
            robot_id = objspec["id"]
        if objspec["class"] not in idbyclass:
            idbyclass[objspec["class"]] = set()
        idbyclass[objspec["class"]].add(objspec["id"])
        id2objects[obj.id] = obj

    target_object = None
    if "target_id" in spec:
        target_id = spec["target_id"]
        target_object = (target_id, id2objects[target_id]["class"])

    detectors = []
    for i in range(len(spec["detectors"])):
        # Build a RangeDetector
        dspec = spec["detectors"][i]
        sensors = {}
        for ref in dspec["sensors"]:
            # ref can be either an object id or a classname.
            # If classname, then all objects with that class will
            # be paired with this sensor.
            if type(ref) == int:
                objids = [ref]
            else:
                # ref is taken as object class
                objids = idbyclass[ref]
            for objid in objids:
                sensor_spec = dspec["sensors"][ref]
                sensors[objid] = parse_sensor(sensor_spec)

        params = {}
        for param_name in dspec["params"]:
            pspec = dspec["params"][param_name]
            params[param_name] = {}
            if type(pspec) == dict:
                for ref in pspec:
                    if type(ref) == int:
                        objids = [ref]
                    else:
                        objids = idbyclass[ref]
                    for objid in objids:
                        if objid not in pspec:
                            print("Warning: Parameter %s unspecified for object %d"\
                                  % (param_name, objid))
                            continue
                        params[param_name][objid] = pspec[ref]
        detectors.append(RangeDetector(dspec["id"], robot_id,
                                       dspec["type"], sensors,
                                       energy_cost=dspec.get("energy_cost", 0),
                                       name=dspec["name"],
                                       locations=locations,
                                       objects=objects,
                                       **params))
    return dict(dim=dim, name=name,
                robot_id=robot_id,
                objects=objects, detectors=detectors,
                target_object=target_object,
                idbyclass=idbyclass,
                id2objects=id2objects,
                locations=locations)

def parse_dist(domain_info, spec):
    """Build a joint distribution given spec (dict)

    The spec for distribution is a list of dictionaries,
    where each specifies a factor either between objects
    or at the class level. The factors at the class
    level will be interpreted as factors between every
    possible combination of the instances of the classes.
    So each factor specification either specifies one
    or multiple probability distributions between a
    set of objects. The distribution is specified by "dist",
    which is interpreted as a spatial relation (or, it could
    be about a single object, such as 'uniform'). The string
    of the spatial relation should correspond to one of the
    functions in `spatial_relation.py`. The parameters to
    initialize these distributions are given in `params`.
    See config/simple_config.yaml for an example.
    """
    factors = []
    variables = set()
    for i in range(len(spec["probability"])):
        dist_spec = spec["probability"][i]
        spatial_relation = eval(dist_spec["dist"])
        params = dist_spec.get("params", {})
        locations = domain_info["locations"]

        # objids is a 2D array:
        # [
        #    (class1) [ objid ...]
        #    (class2) [ objid ...]
        # ]
        if "objects" in dist_spec:
            objids = [dist_spec["objects"]]
            objid_combos = objids
            # Do a cartesian product of the locations for each object
            locations_combos = itertools.product(*([locations]*len(dist_spec["objects"])))

        elif "classes" in dist_spec:
            objids = []
            for cls in dist_spec["classes"]:
                objids.append(domain_info["idbyclass"][cls])
            # Obtain the cartesian product of the object ids of classes
            objid_combos = itertools.product(*objids)
            # Do a cartesian product of the locations for each object in combo
            locations_combos = itertools.product(*([locations]*len(objids)))

        else:
            raise ValueError("No object specified for distribution (item %d)" % i)

        for combo in objid_combos:
            # combo: a tuple of object ids
            variables.update(set(svar(objid) for objid in combo))
            weights = [] # location to weight
            for location_combo in locations_combos:
                # combo: a tuple of locations
                # Create an entry for this location combo in the factor table
                setting = []
                for i in range(len(location_combo)):
                    objid = combo[i]
                    objclass = domain_info["id2objects"][objid]["class"]
                    loc = location_combo[i]
                    objstate = LocObjState(objid, objclass, {"loc": loc})
                    setting.append((svar(objid), objstate))
                prob = spatial_relation(*location_combo, **params)
                weights.append((setting, prob))
            factor = TabularDistribution([svar(objid) for objid in combo],
                                         weights)
            factors.append(factor)
    factor_graph = FactorGraph(list(sorted(variables)),
                               factors)
    return factor_graph

def parse_robot(info, spec):
    # Move actions
    actions = set()
    move_spec = spec["robot"]["move"]
    for i in range(len(move_spec)):
        delta = move_spec[i]["delta"]
        for j in range(len(delta)):
            if type(delta[j]) == str:
                delta[j] = eval(delta[j])
        move_action = Move(delta, name=move_spec[i]["name"],
                           energy_cost=move_spec[i]["energy_cost"])
        actions.add(move_action)

    # Declare actions
    declare_type = spec["robot"]["declare"]
    if declare_type == "on_top":
        actions.add(Declare())
    elif declare_type == "all":
        for loc in info["locations"]:
            actions.add(Declare(loc=loc))
    else:
        raise ValueError("Unsupported declare type %s" % declare_type)

    # Detector actions
    for detector in info["detectors"]:
        actions.add(UseDetector(detector.id,
                                name=detector.name,
                                energy_cost=detector.energy_cost))

    if spec["robot"]["transition"] == "deterministic":
        robot_trans = DetRobotTrans(info["robot_id"],
                                    locations=info["locations"],
                                    schema=spec["robot"]["move_schema"],
                                    actions=actions)
    else:
        raise ValueError("Unsupported robot transition type: %s" % spec["transition"])

    return actions, robot_trans


def read_domain_file(domain_file):
    with open(domain_file) as f:
        spec = yaml.load(f, Loader=yaml.Loader)
    return spec

def problem_from_file(domain_file):
    return problem_parser(read_domain_file(domain_file))

def problem_parser(spec, joint_dist=None, joint_dist_path=None):
    """
    Returns a Field2D search problem by parsing the domain file.
    First, the domain file (a .yaml) file will be read.

    Args:
        joint_dist (JointDist or None): In some cases we already
            have precomputed the joint distribution, independent
            from the given spec. So we will just use the given one,
            to build the Problem, if it is not None.
        joint_dist_path (str): Path to a .pkl file that stores the joint distribution.
    """
    info = parse_domain(spec)

    # parse the joint distribution
    if joint_dist_path is not None:
        print("Loading joint distribution from %s" % joint_dist_path)
        with open(joint_dist_path, "rb") as f:
            joint_dist = pickle.load(f)
    if joint_dist is None:
        joint_dist = parse_dist(info, spec)

    # parse the robot
    actions, robot_trans = parse_robot(info, spec)
    robot_model = RobotModel(info["detectors"],
                             actions,
                             robot_trans)
    problem = Field2D(info["dim"], info["objects"], joint_dist,
                      info["robot_id"], robot_model,
                      locations=info["locations"],
                      target_object=info.get("target_object", None))
    return problem


if __name__ == "__main__":
    problem_parser("./configs/simple_config.yaml")
