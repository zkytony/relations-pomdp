import pomdp_py
import random
import yaml
import pickle
import copy
from pprint import pprint
from corrsearch.experiments.domains.thor.problem import *
from corrsearch.experiments.domains.thor.thor import *
from corrsearch.experiments.domains.thor.test_trial import *
from sciex import Experiment
from datetime import datetime as dt
import shutil

def fill_detector(detector_id, name, cls, cfg):
    detector = dict(
        name=name,
        id=detector_id,
        type="loc",
        energy_cost=cfg["energy_cost"],
        sensors={
            cls:dict(
                type="fan",
                params=dict(
                    fov=cfg["fov"],
                    min_range=0.0,
                    max_range=cfg["max_range"]
                )
            )
        },
        params=dict(
            true_positive={cls: cfg["truepos"]},
            false_positive={cls: 0.0},
            sigma={cls: 0.1}
        )
    )
    return detector


def fill_dist(objclass, target_class, cfg):
    dist = dict(
        classes=[objclass, target_class],
        dist=cfg["rel"],
        params=dict(radius=cfg["radius"])
    )
    return dist

grid_size = 0.4
ntrials = 15
max_steps = 50
split = 5

OUTPUT_DIR = os.path.join("results", "thor")
exp_name = "ThorSearchHH-GridSize{}".format(grid_size)
start_time_str = dt.now().strftime("%Y%m%d%H%M%S%f")[:-3]
exp_name += "_" + start_time_str

os.makedirs(os.path.join(OUTPUT_DIR, exp_name), exist_ok=True)
shutil.copytree("domains/thor/data", os.path.join(OUTPUT_DIR, exp_name, "data"))
shutil.copytree("domains/thor/config", os.path.join(OUTPUT_DIR, exp_name, "config"))

case1_kitchen = {
    "scene": "FloorPlan1",
    "scene_type": "kitchen",
    "objects":
    (("PepperShaker", dict(fov=90, max_range=0.8, truepos=0.5, energy_cost=4.0)),
     ("StoveBurner", dict(rel="nearby", radius=0.8, fov=90, max_range=1.6, truepos=0.9, energy_cost=1.0))),
     # ("Mug", dict(rel="nearby", radius=0.9, fov=80, max_range=1.0, truepos=0.95)))
}

case2_kitchen = {
    "scene": "FloorPlan1",
    "scene_type": "kitchen",
    "objects":
    (("Statue", dict(fov=90, max_range=0.8, truepos=0.7, energy_cost=3.0)),
     ("Shelf", dict(rel="nearby", radius=0.8, fov=90, max_range=1.2, truepos=0.9, energy_cost=0.5))),
}


# case3_living = {
#     "scene": "FloorPlan201",
#     "scene_type": "living#room",
#     "objects":
#     (("KeyChain", dict(fov=90, max_range=0.8, truepos=0.5, energy_cost=3.0)),
#      ("Vase", dict(rel="nearby", radius=0.8, fov=90, max_range=1.2, truepos=0.9, energy_cost=1.0)))
#      # ("Book", dict(rel="nearby", radius=0.9, fov=80, max_range=0.9, truepos=0.95)))
# }


# case4_living = {
#     "scene": "FloorPlan201",
#     "scene_type": "living#room",
#     "objects":
#     (("Book", dict(fov=90, max_range=0.8, truepos=0.8, energy_cost=3.0)),
#      ("DiningTable", dict(rel="nearby", radius=0.8, fov=90, max_range=1.6, truepos=0.9, energy_cost=1.0))),
#      # ("Book", dict(rel="nearby", radius=0.9, fov=80, max_range=0.9, truepos=0.95)))
# }

cases = [
    case1_kitchen,
    case2_kitchen,
    # case3_living,
    # case4_living,
]

all_trials = []
for case in cases:

    # Build spec
    spec = {}
    spec["scene_name"] = case["scene"]
    spec["scene_type"] = case["scene_type"]
    spec["grid_size"] = grid_size
    spec["boundary_thickness"] = 4

    spec["robot_id"] = 0
    spec["move_schema"] = "topo"
    spec["rotate_actions"] = [
        dict(
            name = "left",
            delta = [0.0, "-math.pi/2"],
            energy_cost = 0.0
        ),
        dict(
            name = "right",
            delta = [0.0, "math.pi/2"],
            energy_cost = 0.0
        )
    ]

    spec["object_classes"] = []
    spec["detectors"] = []
    spec["probability"] = []

    target_objtup = case["objects"][0]
    target_class, target_cfg = target_objtup
    spec["object_classes"].append(target_class)
    spec["target_class"] = target_class

    # detector
    count = 100
    detector = fill_detector(count,
                             "target-{}-detector".format(target_class),
                             target_class,
                             target_cfg)
    spec["detectors"].append(detector)
    spec_target_only = copy.deepcopy(spec)
    spec_target_only["probability"].append(dict(classes=[target_class], dist="uniform"))

    os.makedirs(os.path.join(OUTPUT_DIR, exp_name, "resources"), exist_ok=True)

    for other_objtup in case["objects"][1:]:
        count += 100
        objclass, objcfg = other_objtup
        detector = fill_detector(count,
                                 "{}-detector".format(objclass),
                                 objclass,
                                 objcfg)
        spec["detectors"].append(detector)

        pairdist = fill_dist(objclass, target_class, objcfg)
        spec["probability"].append(dict(classes=[objclass], dist="uniform"))
        spec["probability"].append(pairdist)

    joint_dist_file = "{}-{}-{}-GridSize{}_joint-dist.pkl".format(spec["scene_type"], spec["scene_name"],
                                                                  "_".join(spec["object_classes"]),
                                                                  grid_size)
    spec["joint_dist_path"] = os.path.join("resources", joint_dist_file)
    spec_target_only["joint_dist_path"] = None  # we don't expect this to be needed
    exp_resources_path = os.path.join(OUTPUT_DIR, exp_name, "resources")
    os.makedirs(exp_resources_path, exist_ok=True)
    if not os.path.exists(os.path.join(exp_resources_path, joint_dist_file)):
        # We will instantiate the problem for once and save its joint distribution.
        problem = ThorSearch.parse(spec, scene_data_path="./domains/thor/data",
                                   topo_dir_path="./domains/thor/data/topo")
        with open(os.path.join(exp_resources_path, joint_dist_file), "wb") as f:
            pickle.dump(problem.joint_dist, f)
        problem.env.controller.stop()

    print("Built spec")
    pprint(spec)

    # Make config
    for i in range(ntrials):
        ########### HEURISTIC
        trial_name = "{}-{}-{}_{}_heuristic#noprune#iq"\
                     .format(spec["scene_type"], spec["target_class"],
                             spec["scene_name"].replace("_", "#"),
                             i+1)
        planner_config = copy.deepcopy(HEURISTIC_ONLINE_PLANNER_CONFIG)
        instance_config = {"topo_move_cost_factor": 4.0,
                           "applies_all_everywhere": False}
        config_corr = make_config(copy.deepcopy(spec),
                                  init_belief="prior",
                                  planner="HeuristicSequentialPlanner",
                                  planner_config=planner_config,
                                  instance_config=instance_config,
                                  max_steps=max_steps)
        trial = make_trial(config_corr, trial_name)
        all_trials.append(trial)

        ########### HEURISTIC (Applies All Everywhere)
        trial_name = "{}-{}-{}_{}_heuristic#noprune#iq-all-everywhere"\
                     .format(spec["scene_type"], spec["target_class"],
                             spec["scene_name"].replace("_", "#"),
                             i+1)
        planner_config = copy.deepcopy(HEURISTIC_ONLINE_PLANNER_CONFIG)
        instance_config = {"topo_move_cost_factor": 4.0,
                           "applies_all_everywhere": True}
        config_corr = make_config(copy.deepcopy(spec),
                                  init_belief="prior",
                                  planner="HeuristicSequentialPlanner",
                                  planner_config=planner_config,
                                  instance_config=instance_config,
                                  max_steps=max_steps)
        trial = make_trial(config_corr, trial_name)
        all_trials.append(trial)

        ########## TARGET ONLY (Applies All Everywhere)
        trial_name = "{}-{}-{}_{}_target-only-heuristic-all-everywhere"\
                     .format(spec["scene_type"], spec["target_class"],
                             spec["scene_name"].replace("_", "#"),
                             i+1)
        planner_config = copy.deepcopy(HEURISTIC_ONLINE_PLANNER_CONFIG)
        instance_config = {"topo_move_cost_factor": 4.0,
                           "applies_all_everywhere": True}
        config_target_only = make_config(spec_target_only,
                                         init_belief="uniform",
                                         planner="HeuristicSequentialPlanner",
                                         planner_config=planner_config,
                                         instance_config=instance_config,
                                         max_steps=max_steps)
        trial = make_trial(config_target_only, trial_name)
        all_trials.append(trial)


random.shuffle(all_trials)
exp = Experiment(exp_name, all_trials, OUTPUT_DIR, verbose=True,
                 add_timestamp=False)
exp.generate_trial_scripts(split=split)
print("Trials generated at %s/%s" % (exp._outdir, exp.name))
print("Find multiple computers to run these experiments.")
