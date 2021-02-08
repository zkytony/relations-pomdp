import pomdp_py
import random
from corrsearch.experiments.domains.field2d.problem import *
from corrsearch.experiments.domains.field2d.parser import *
from corrsearch.experiments.trial import SearchTrial
from corrsearch.models import *
from corrsearch.objects import *


def make_trial(domain_file):
    random.seed(500)
    problem_creator = "corrsearch.experiments.domains.field2d.parser.problem_parser"
    problem_config = {"domain_file": domain_file}

    instance_config = dict(
        init_locs="random",
        init_robot_setting=((0, 0, 0), 100),
        init_belief="informed"
    )
    planner = "pomdp_py.POUCT"
    planner_init_config = dict(
        max_depth=10,
        discount_factor=0.95,
        num_sims=100,
        exploration_const=200
    )
    planner_exec_config = dict()

    visualize = True
    visualize_config = dict(
        res=30
    )

    exec_config = dict(
        max_steps=10
    )
    discount_factor = 0.95

    config = dict(
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
    trial = SearchTrial("test_trial", config)
    return trial

if __name__ == "__main__":
    trial = make_trial("./configs/simple_config.yaml")
    trial.run()
