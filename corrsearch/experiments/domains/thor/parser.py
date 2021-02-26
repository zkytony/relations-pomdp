"""
Config parser for THOR.

Very similar to the parser for Field2D but differ in details.
"""
from corrsearch.experiments.domains.thor.detector import *
from corrsearch.experiments.domains.thor.spatial_corr import *
from corrsearch.probability import *
import numpy as np
import math
import yaml
import pickle
import os

def parse_sensor(sensor_spec):
    """Build sensor given sensor_space (dict)"""
    if sensor_spec["type"] == "fan":
        sensor = FanSensorThor(**sensor_spec["params"])
    else:
        raise ValueError("Unrecognized sensor type %s" % sensor_spec["type"])
    return sensor

def parse_detectors(scene_info, spec_or_filepath, robot_id):
    """
    scene_info (dict) see load_scene_info in process_scenes.py
    """
    if type(spec_or_filepath) == str:
        with open(spec_or_filepath) as f:
            spec_detectors = yaml.load(f)
    elif type(spec_or_filepath) == list:
        spec_detectors = spec_or_filepath
    else:
        raise TypeError("spec_or_filepath must be a string or a dict.")

    detectors = []
    for dspec in spec_detectors:
        sensors = {}
        for ref in dspec["sensors"]:
            assert type(ref) == str, "THOR sensors should be specified at type level"
            objtype = ref
            objid_for_type = scene_info.objid_for_type(objtype)

            sensor_spec = dspec["sensors"][ref]
            sensors[objid_for_type] = parse_sensor(sensor_spec)

        params = {}
        for param_name in dspec["params"]:
            pspec = dspec["params"][param_name]
            params[param_name] = {}
            if type(pspec) == dict:
                for ref in pspec:
                    assert type(ref) == str, "THOR detector params should be specified at type level"
                    objtype = ref
                    objid_for_type = scene_info.objid_for_type(objtype)
                    params[param_name][objid_for_type] = pspec[ref]
        detector = RangeDetector(dspec["id"], robot_id,
                                 dspec["type"], sensors,
                                 energy_cost=dspec.get("energy_cost", 0),
                                 name=dspec["name"],
                                 **params)
        detectors.append(detector)
    return detectors

def parse_dist(scene_info, grid_map, thor_locations, prob_spec, grid_size=0.25):
    """
    grid_map should be the one loaded directly from the scene.

    thor_locations should be possible object locations in thor coordinates (i.e. meters).
    """
    factors = []
    variables = set()
    for tt, dist_spec in enumerate(prob_spec):
        spatial_relation = eval(dist_spec["dist"])
        params = dist_spec.get("params", {})
        print("Parsing distribution spec {}: {}, {}"\
              .format(tt, dist_spec["classes"], dist_spec["dist"]))

        objids = []
        for cls in dist_spec["classes"]:
            objids.append([scene_info.objid_for_type(cls)])
            if len(scene_info.pomdp_objids(cls)) > 1:
                print("WARNING: {} has multiple instances."\
                      "Only one instance considered in the factor graph.".format(cls))

        # Obtain the cartesian product of the object ids of classes
        objid_combos = itertools.product(*objids)
        # Do a cartesian product of the locations for each object in combo
        thor_location_combos = itertools.product(*([thor_locations]*len(objids)))

        for combo in objid_combos:
            variables.update(set(svar(objid) for objid in combo))
            weights = []
            settings = set()
            for thor_location_combo in thor_location_combos:
                setting = []
                for i in range(len(thor_location_combo)):
                    objid = combo[i]
                    objclass = scene_info.obj_type(objid)
                    loc = grid_map.to_grid_pos(*thor_location_combo[i],
                                               grid_size=grid_size)
                    objstate = LocObjState(objid, objclass, {"loc": loc})
                    setting.append((svar(objid), objstate))
                if tuple(setting) not in settings:
                    prob = spatial_relation(*thor_location_combo, **params)
                    weights.append((setting, prob))
                    settings.add(tuple(setting))
            factor = TabularDistribution([svar(objid) for objid in combo],
                                         weights)
            for var in factor.variables:
                print("   {} valrange size: {}".format(var, len(factor.valrange(var))))

            factors.append(factor)

    factor_graph = FactorGraph(list(sorted(variables)), factors)

    return factor_graph


def parse_move_actions(move_spec):
    # Move actions
    move_actions = set()
    for i in range(len(move_spec)):
        delta = move_spec[i]["delta"]
        for j in range(len(delta)):
            if type(delta[j]) == str:
                delta[j] = eval(delta[j])
        move_action = Move(delta, name=move_spec[i]["name"],
                           energy_cost=move_spec[i]["energy_cost"])
        move_actions.add(move_action)
    return move_actions



def TEST():
    with open("./config/config-FloorPlan_Train1_1-Laptop-simple.yaml") as f:
        spec = yaml.load(f, Loader=yaml.Loader)

    problem = ThorSearch.parse(spec)
