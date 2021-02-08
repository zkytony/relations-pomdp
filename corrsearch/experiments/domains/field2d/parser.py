import yaml
import pomdp_py
import itertools
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
    Care about "objects", "detectors", "name", "bg", "dim"
    Args:
        spec (dict). Contains
    Return:
        Field2D problem, so that Field2D can be created and POMDPs can be instantiated.
    """
    dim = spec["dim"]
    name = spec["name"]

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
                                       **params))
    return dict(dim=dim, name=name,
                robot_id=robot_id,
                objects=objects, detectors=detectors,
                target_object=target_object,
                idbyclass=idbyclass,
                id2objects=id2objects)

def parse_dist(domain_info, spec):
    """Build a joint distribution given spec (dict)"""
    factors = []
    variables = set()
    for i in range(len(spec["probability"])):
        dist_spec = spec["probability"][i]
        spatial_relation = eval(dist_spec["dist"])
        params = dist_spec.get("params", {})

        # objids is a 2D array:
        # [
        #    (class1) [ objid ...]
        #    (class2) [ objid ...]
        # ]
        if "objects" in dist_spec:
            objids = [dist_spec["objects"]]

        elif "classes" in dist_spec:
            objids = []
            for cls in dist_spec["classes"]:
                objids.append(domain_info["idbyclass"][cls])

        else:
            raise ValueError("No object specified for distribution (item %d)" % i)

        # Obtain the cartesian product of the object ids of classes
        objid_combos = itertools.product(*objids)
        # Do a cartesian product of the locations for each object in combo
        locations = domain_info["locations"]
        locations_combos = itertools.product(*([locations]*len(objids)))
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
        robot_trans = DetRobotTrans(info["robot_id"], spec["robot"]["move_schema"])
    else:
        raise ValueError("Unsupported robot transition type: %s" % spec["transition"])

    return actions, robot_trans


def problem_parser(domain_file):
    with open(domain_file) as f:
        spec = yaml.load(f, Loader=yaml.Loader)
    info = parse_domain(spec)

    # enumerate list of locations
    dim = info["dim"]
    locations = [(x,y) for x in range(dim[0]) for y in range(dim[1])]
    info["locations"] = locations

    # parse the joint distribution
    joint_dist = parse_dist(info, spec)

    # parse the robot
    actions, robot_trans = parse_robot(info, spec)
    robot_model = RobotModel(info["detectors"],
                             actions,
                             robot_trans)
    problem = Field2D(dim, info["objects"], joint_dist,
                      info["robot_id"], robot_model,
                      locations=locations,
                      target_object=info.get("target_object", None))
    return problem


if __name__ == "__main__":
    problem_parser("./configs/simple_config.yaml")
