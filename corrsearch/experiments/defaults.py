# planner configurations
POMCP_PLANNER_CONFIG = {
    "max_depth": 25,
    "discount_factor": 0.95,
    "num_sims": 1000,
    "exploration_const": 200
}

ENTROPY_PLANNER_CONFIG = {
    "declare_threshold": 0.9,
    "entropy_improvement_threshold": 1e-3,
    "num_samples": 100
}

RANDOM_PLANNER_CONFIG = {
    "declare_threshold": 0.9,
}

HEURISTIC_ONLINE_PLANNER_CONFIG = {
    **POMCP_PLANNER_CONFIG,
    "k": 2,
    "num_zsamples": 30,
    "num_rsamples": 30,
    "gamma": 0.95,
    "num_visits_init": 0,
    "init_qvalue_lower_bound": False
}


########## FUNCTIONS FOR FIELD2D SPEC BUILDING ###########
def detid(objid):
    return objid*100

# functions for building domain spec
def add_object(spec, objid, objclass, color, dim=[1,1]):
    if "objects" not in spec:
        spec["objects"] = []

    spec["objects"].append({"class": objclass,
                            "id": int(objid),
                            "color": list(color),
                            "dim": list(dim)})

def set_target(spec, target_id):
    spec["target_id"] = target_id

def set_dim(spec, dim):
    """Search space dimensions"""
    assert len(dim) == 2, "Dim must be 2D"
    spec["dim"] = list(dim)

def add_disk_sensor(dspec, objid, radius, true_positive, false_positive=0.0,
                    sigma=0.1):
    if "sensors" not in dspec:
        dspec["sensors"] = {}
    if "params" not in dspec:
        dspec["params"] = {}
    if "true_positive" not in dspec:
        dspec["params"]["true_positive"] = {}
    if "false_positive" not in dspec:
        dspec["params"]["false_positive"] = {}
    if "sigma_positive" not in dspec:
        dspec["params"]["sigma"] = {}
    dspec["sensors"][objid] = {"type": "disk",
                               "params": {"radius": radius}}
    dspec["params"]["true_positive"][objid] = true_positive
    dspec["params"]["false_positive"][objid] = false_positive
    dspec["params"]["sigma"][objid] = sigma

def add_fan_sensor(dspec, objid, fov, min_range, max_range, true_positive,
                   false_positive=0.0, sigma=0.1):
    if "sensors" not in dspec:
        dspec["sensors"] = {}
    if "params" not in dspec:
        dspec["params"] = {}
    if "true_positive" not in dspec:
        dspec["params"]["true_positive"] = {}
    if "false_positive" not in dspec:
        dspec["params"]["false_positive"] = {}
    if "sigma_positive" not in dspec:
        dspec["params"]["sigma"] = {}
    dspec["sensors"][objid] = {"type": "disk",
                               "params": {"fov": fov,
                                          "min_range": min_range,
                                          "max_range": max_range}}
    dspec["params"]["true_positive"][objid] = true_positive
    dspec["params"]["false_positive"][objid] = false_positive
    dspec["params"]["sigma"][objid] = sigma

def add_detector(spec, name, detector_id, detector_type, energy_cost=0.0):
    """Example detector_type: loc, label"""
    if "detectors" not in spec:
        spec["detectors"] = []
    spec["detectors"].append({"name": name,
                              "id": detector_id,
                              "type": detector_type,
                              "energy_cost": energy_cost,
                              "sensors": {},
                              "params": {}})
    return spec["detectors"][-1]

def remove_detector(spec, detector_id):
    if "detectors" not in spec:
        return
    dindex = None
    for i, dspec in enumerate(spec["detectors"]):
        if dspec["id"] == detector_id:
            dindex = i
            break
    if dindex is None:
        raise ValueError("Detector %d was not added" % detector_id)
    spec["detectors"].pop(i)

def get_detector(spec, detector_id):
    if "detectors" not in spec:
        raise ValueError("No detector in spec")
    for i, dspec in enumerate(spec["detectors"]):
        if dspec["id"] == detector_id:
            return dspec
    raise ValueError("Detector {} not in spec".format(detector_id))


def add_factor(spec, dist_type, params={}, objects=None, classes=None):
    if "probability" not in spec:
        spec["probability"] = []

    fspec = {}
    if objects is None and classes is None:
        raise ValueError("Either supply a list of object ids or classes")
    if objects is not None:
        fspec["objects"] = objects
    else:
        fspec["classes"] = classes
    fspec["dist"] = dist_type
    fspec["params"] = params
    spec["probability"].append(fspec)

def add_robot_simple2d(spec):
    spec["robot"] = {
          "move_schema": "xy",
          "move": [
              {"name": "north",
               "delta": [0, -1, "3*math.pi/2"],
               "energy_cost": 0.0},

              {"name": "east",
               "delta": [1, 0, 0],
               "energy_cost": 0.0},

              {"name": "south",
               "delta": [0, 1, "math.pi/2"],
               "energy_cost": 0.0},

              {"name": "west",
               "delta": [-1, 0, "math.pi"],
               "energy_cost": 0.0}],
        "declare": "on_top",
        "transition": "deterministic"
    }


# Predefined objects
OBJECTS = [
    (1, "blue-cube", [30, 30, 200]),
    (2, "red-cube", [232, 55, 35]),
    (3, "orange-cube", [252, 178, 50]),
    (4, "purple-cube", [168, 50, 252]),
    (5, "gray-cube", [136, 134, 138]),
]
