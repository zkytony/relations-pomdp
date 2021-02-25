import pomdp_py
import random
import yaml
from corrsearch.experiments.domains.thor.problem import *
from corrsearch.experiments.domains.thor.process_scenes import load_scene_info
from corrsearch.experiments.trial import SearchTrial
from corrsearch.experiments.defaults import *
from corrsearch.models import *
from corrsearch.objects import *

def make_config(scene_name,
                target_type,
                detector_spec_path,
                move_actions=MOVE_ACTIONS,
                grid_size=0.25,
                init_belief="uniform",
                planner_config={},
                planner="pomdp_py.POUCT",
                max_steps=100,
                visualize=True,
                seed=None,
                step_delay=0.1,
                viz_res=30):

    scene_info = load_scene_info(scene_name)
    robot_id = 0
    target_id = min(scene_info.pomdp_objids(target_type))

    problem_creator = "ThorSearch"
    problem_config = dict(
        robot_id=robot_id,
        target_object=(target_id, target_type),
        scene_name=scene_name,
        scene_info=scene_info,
        detectors_spec_path=detector_spec_path,
        grid_size=grid_size
    )
    planner_exec_config = dict()

    instance_config = dict(
        init_belief=init_belief
    )

    if planner == "pomdp_py.POUCT":
        planner_init_config = dict(
            max_depth=planner_config.get("max_depth", 10),
            discount_factor=planner_config.get("discount_factor", 0.95),
            num_sims=planner_config.get("num_sims", 200),
            exploration_const=planner_config.get("exploration_const", 200)
        )
    else:
        planner_init_config = planner_config

    visualize_config = dict(
        res=viz_res
    )

    exec_config = dict(
        max_steps=max_steps,
        step_delay=step_delay,
        debugging=True
    )
    discount_factor = planner_config.get("discount_factor", 0.95)

    # modules that need to be imported when a trial runs
    imports = [
        "from corrsearch.experiments.domains.thor.problem import *",
        "from corrsearch.experiments.domains.thor.thor import *",
        "from corrsearch.planning import *"
    ]
    config = dict(
        imports=imports,
        problem_creator=problem_creator,
        problem_config=problem_config,
        instance_config=instance_config,
        planner=planner,
        planner_init_config=planner_init_config,
        planner_exec_config=planner_exec_config,
        visualize_config=visualize_config,
        exec_config=exec_config,
        discount_factor=discount_factor,
        visualize=visualize
    )
    return config

def make_trial(config, trial_name="test_trial"):
    trial = SearchTrial(trial_name, config)
    return trial


if __name__ == "__main__":
    config = make_config("FloorPlan_Train1_1",
                         "Apple",
                         "./config/detectors_spec.yaml",
                         planner="pomdp_py.POUCT",
                         # "RandomPlanner",
                         # planner="HeuristicSequentialPlanner",#"EntropyMinimizationPlanner",
                         grid_size=0.25,
                         planner_config=POMCP_PLANNER_CONFIG,#HEURISTIC_ONLINE_PLANNER_CONFIG,#RANDOM_PLANNER_CONFIG,#,#RANDOM_PLANNER_CONFIG,
                         #ENTROPY_PLANNER_CONFIG,#
                         seed=100,
                         init_belief="informed")
    trial = make_trial(config)
    trial.run()
