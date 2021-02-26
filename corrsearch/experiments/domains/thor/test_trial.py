import pomdp_py
import random
import yaml
from corrsearch.experiments.domains.thor.problem import *
from corrsearch.experiments.domains.thor.thor import load_scene_info
from corrsearch.experiments.trial import SearchTrial
from corrsearch.experiments.defaults import *
from corrsearch.models import *
from corrsearch.objects import *

def make_config(spec_path,
                init_belief="uniform",
                planner_config={},
                planner="pomdp_py.POUCT",
                max_steps=100,
                visualize=True,
                seed=None,
                step_delay=0.1,
                viz_res=30):
    with open(spec_path) as f:
        spec = yaml.load(f, Loader=yaml.Loader)
    problem_creator = "ThorSearch.parse"
    problem_config = dict(
        spec_or_path=spec
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
        step_delay=step_delay
        # debugging=True
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
    config = make_config("./config/config-FloorPlan1-Mug-simple.yaml",
                         planner="HeuristicSequentialPlanner",#"EntropyMinimizationPlanner",#"HeuristicSequentialPlanner",#"RandomPlanner",#"pomdp_py.POUCT",
                         # "RandomPlanner",
                         # planner="HeuristicSequentialPlanner",#"EntropyMinimizationPlanner",
                         planner_config=HEURISTIC_ONLINE_PLANNER_CONFIG,#RANDOM_PLANNER_CONFIG,#POMCP_PLANNER_CONFIG,#HEURISTIC_ONLINE_PLANNER_CONFIG,#RANDOM_PLANNER_CONFIG,#,#RANDOM_PLANNER_CONFIG,
                         #ENTROPY_PLANNER_CONFIG,#
                         seed=100,
                         init_belief="uniform")
    trial = make_trial(config)
    trial.run()
