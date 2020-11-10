from sciex import Experiment, Trial, Event, Result,\
    YamlResult, PklResult, PostProcessingResult

class StatesResult(PklResult):
    def __init__(self, states):
        """list of state objects"""
        super().__init__(states)

    @classmethod
    def FILENAME(cls):
        return "states.pkl"

    @classmethod
    def gather(cls, results):
        """`results` is a mapping from specific_name to a dictionary {seed: actual_result}.
        Returns a more understandable interpretation of these results"""
        pass

    @classmethod
    def save_gathered_results(cls, gathered_results, path):
        pass
