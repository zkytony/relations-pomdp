import pomdp_py
import argparse
import copy
from corrsearch.experiments.domains.field2d.problem import *
from corrsearch.experiments.domains.field2d.parser import *
from corrsearch.experiments.trial import SearchTrial

class PlaybackPlanner(pomdp_py.Planner):
    def __init__(self, history):
        self._step = 0
        self.history = history

    def plan(self, agent):
        action = self.history[self._step][0]
        self._step += 1
        return action

def main():
    parser = argparse.ArgumentParser(description="replay a trial")
    parser.add_argument("trial_path", type=str, help="Path to trial directory")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--delay", type=float, help="delay in seconds between two steps", default=0.5)
    args = parser.parse_args()

    with open(os.path.join(args.trial_path, "trial.pkl"), "rb") as f:
        trial = pickle.load(f)

    with open(os.path.join(args.trial_path, "states.pkl"), "rb") as f:
        states = pickle.load(f)

    with open(os.path.join(args.trial_path, "history.pkl"), "rb") as f:
        history = pickle.load(f)

    config = copy.deepcopy(trial.config)
    config["planner"] = "PlaybackPlanner"
    config["planner_init_config"] = {"history": history}
    config["visualize"] = True
    config["imports"].append("from corrsearch.experiments.replay import PlaybackPlanner")
    config["exec_config"]["step_delay"] = args.delay
    trial = SearchTrial(trial.name, config, verbose=True)
    trial.run()

if __name__ == "__main__":
    main()
