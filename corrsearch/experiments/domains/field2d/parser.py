import yaml
import pomdp_py
from corrsearch.models.problem import *
from corrsearch.objects.template import Object
from corrsearch.experiments.domains.field2d.detector import *
from corrsearch.experiments.domains.field2d.problem import Field2D

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
    bg = spec["bg"]

    objects = []
    robot_id = None
    idbyclass = {}
    for i in range(len(spec["objects"])):
        objspec = spec["objects"][i]
        obj = Object(objspec["id"], objspec)
        objects.append(obj)
        if objspec["class"] == "robot":
            robot_id = objspec["id"]
        if objspec["class"] not in idbyclass:
            idbyclass[objspec["class"]] = set()
        idbyclass[objspec["class"]].add(objspec["id"])

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
                                       dspec["type"], sensors, **params))
    return dict(dim=dim, name=name, bg=bg,
                robot_id=robot_id,
                objects=objects, detectors=detectors,
                idbyclass=idbyclass)

def parse_dist(domain_info, spec):
    """Build a joint distribution given spec (dict)"""
    dim = domain_info["dim"]
    locations = [(x,y) for x in range(dim[0]) for y in range(dim[1])]
    for i in range(len(spec["probability"])):
        dist_spec = spec["probability"]
        if "object" in dist_spec:
            objids = [dist_spec["object"]]
        elif "class" in dist_spec:
            objids = domain_info["idbyclass"][dist_spec["class"]]
        else:
            raise ValueError("No object specified for distribution (item %d)" % i)

        cond_objids = None
        if "condition" in dist_spec:
            if type(dist_spec["condition"]) == int:
                cond_objids = [dist_spec["condition"]]
            else:
                cond_objids = domain_info["idbyclass"][dist_spec["condition"]]

        # Build probability table




    pass

def problem_parser(domain_file):
    with open(domain_file) as f:
        spec = yaml.load(f, Loader=yaml.Loader)
    info = parse_domain(spec)
    joint_dist = parse_dist(info, spec)
    return Field2D(dim, name=name, bg=bg,
                   robot_id=robot_id, objects=objects,
                   detectors=detectors)
