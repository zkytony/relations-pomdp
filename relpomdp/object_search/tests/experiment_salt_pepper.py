from sciex import *
from relpomdp.object_search.tests.worlds import *
from relpomdp.object_search.tests.trial import *

def generate_experiment():
    # Will have the robot start at bottom right corner.
    # In each case, run the trial 20 times (due to randomness). The baselines are:
    #
    # random: uses no information at all
    # uniform: POUCT with uniform prior
    # prior_only: POUCT with MRF prior but no observation model
    # both: POUCT with MRF prior + observation model
    # both-greedy: Greedy planner with the same prior (greedy works pretty well)
    # informed: POUCT with informed prior
    
    config = {
        "world": salt_pepper_1,
        "world_configs": {"init_robot_pose": (9,0,0)},
        "target_variable": "Salt_pose",
        "max_steps": 100,
        "visualize": True,
        "user_control": False
    }
    trials = []
    
    for baseline in {"random",
                     "uniform",
                     "mrf#prior#pouct",
                     "mrf#both#pouct",
                     "mrf#both#greedy",
                     "informed"}:
        if baseline == "random":
            config["planner_type"] = "random"
            config["prior_type"] = "uniform"
            config["using_mrf_belief_update"] = False
        elif baseline == "uniform":
            config["planner_type"] = "pouct"
            config["prior_type"] = "uniform"
            config["using_mrf_belief_update"] = False
        elif baseline == "informed":
            config["planner_type"] = "pouct"
            config["prior_type"] = "informed"
            config["using_mrf_belief_update"] = False
        elif baseline.startswith("mrf"):
            config["prior_type"] = "mrf"
            if baseline.split("#")[1] == "both":
                config["using_mrf_belief_update"] = True
            else:
                config["using_mrf_belief_update"] = False
            config["planner_type"] = baseline.split("#")[2]

        if config["planner_type"] == "pouct":
            config["planner"] = {
                "max_depth": 40,
                "discount_factor": 0.95,
                "num_sims": 200,
                "exploration_const": 200
            }

        init_robot_pose = tuple(map(str,config["world_configs"]["init_robot_pose"]))
        for i in range(20):
            trial = SingleObjectSearchTrial("salt-pepper-%s_%d_%s" \
                                            % (",".join(init_robot_pose), i, baseline),
                                            config, verbose=True)
            trials.append(trial)

    random.shuffle(trials)
    output_dir = "./results"
    print("Generating experiment")
    exp = Experiment("SaltPepperA", trials, output_dir, verbose=True, add_timestamp=True)
    exp.generate_trial_scripts(split=5, exist_ok=True)
    print("Find multiple computers to run these experiments.")    

if __name__ == "__main__":
    generate_experiment()
