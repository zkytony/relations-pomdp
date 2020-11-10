from sciex import Experiment, Trial, Event, Result,\
    YamlResult, PklResult, PostProcessingResult

class HistoryResult(PklResult):
    def __init__(self, rewards):
        """rewards: a list of reward floats"""
        super().__init__(rewards)

    @classmethod
    def FILENAME(cls):
        return "history.pkl"

    @classmethod
    def gather(cls, results):
        """`results` is a mapping from specific_name to a dictionary {seed: actual_result}.
        Returns a more understandable interpretation of these results"""
        pass

    @classmethod
    def save_gathered_results(cls, gathered_results, path):
        pass
