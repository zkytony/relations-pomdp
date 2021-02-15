"""Each domain should have a test_trial.py
script in which a Trial is built and tested.
In this script there should be a make_trial(config)
function and a make_config(...) function."""

import pomdp_py
import random
import yaml
from corrsearch.experiments.domains.field2d.problem import *
from corrsearch.experiments.domains.field2d.parser import *
from corrsearch.experiments.trial import SearchTrial
from corrsearch.experiments.defaults import *
from corrsearch.models import *
from corrsearch.objects import *
from relpomdp2.constants import SARSOP_PATH


def make_config(domain_file_or_spec,
                init_locs="random",
                init_belief="prior",
                joint_dist_path=None,
                planner_config={},
                planner="pomdp_py.POUCT",
                max_steps=100,
                visualize=True,
                seed=None,
                step_delay=0.1,
                viz_res=30):
    """
    Set `init_locs` to be random_SEED for deterministic instance generation
    """
    if type(domain_file_or_spec) == str:
        spec = read_domain_file(domain_file_or_spec)
    elif type(domain_file_or_spec) == dict:
        spec = domain_file_or_spec
    else:
        raise TypeError("domain_file_or_spec must be string or dict")
    problem_creator = "corrsearch.experiments.domains.field2d.parser.problem_parser"
    problem_config = {"spec": spec,
                      "joint_dist_path": joint_dist_path}

    instance_config = dict(
        init_locs=init_locs,
        init_robot_setting=((0, 0, 0), 0.0),
        init_belief=init_belief,
        seed=seed,
        explicit_enum_states=False,
        rmax=100,
        rmin=-100
    )
    if planner == "pomdp_py.POUCT":
        planner_init_config = dict(
            max_depth=planner_config.get("max_depth", 10),
            discount_factor=planner_config.get("discount_factor", 0.95),
            num_sims=planner_config.get("num_sims", 200),
            exploration_const=planner_config.get("exploration_const", 200)
        )
    elif planner == "pomdp_py.sarsop":
        planner_init_config = dict(
            pomdpsol_path=SARSOP_PATH,
            timeout=planner_config.get("timeout", 30),
            memory=planner_config.get("timeout", 30),
            precision=planner_config.get("precision", 1e-12),
            pomdp_name=spec["name"],
            logfile=None
        )
    else:
        planner_init_config = planner_config

    planner_exec_config = dict()

    visualize_config = dict(
        res=viz_res
    )

    exec_config = dict(
        max_steps=max_steps,
        step_delay=step_delay,
    )
    discount_factor = planner_config.get("discount_factor", 0.95)

    # modules that need to be imported when a trial runs
    imports = [
        "from corrsearch.experiments.domains.field2d.problem import *",
        "from corrsearch.experiments.domains.field2d.parser import *",
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
    # config = make_config("./configs/simple_config.yaml",
    #                      # planner="EntropyMinimizationPlanner",
    #                      planner="HeuristicSequentialPlanner",
    #                      # planner="pomdp_py.POUCT",
    #                      # planner_config=dict(num_sims=1500),
    #                      # planner_config=dict(num_samples=100,
    #                      #                     entropy_improvement_threshold=1e-3),
    #                      seed=100,
    #                      planner_config=HEURISTIC_ONLINE_PLANNER_CONFIG,
    #                      init_belief="informed")

    with open("configs/fromexp/3by3_1.yaml") as f:
        config = yaml.load(f, Loader=yaml.Loader)
    config["visualize"] = True
    config["instance_config"]["init_robot_setting"] = ((0,0,0), 0.0)
    trial = make_trial(config)
    trial.run()
