from sciex import *
import numpy as np
from scipy import stats
import math
import os
import json
import yaml
import copy

#### Actual results for experiments ####
class RewardsResult(YamlResult):
    def __init__(self, rewards):
        """rewards: a list of reward floats"""
        self._rewards = rewards

    @classmethod
    def FILENAME(cls):
        return "rewards.yaml"

    @classmethod
    def gather(cls, results):
        raise NotImplementedError

    @classmethod
    def save_gathered_results(cls, gathered_results, path):
        raise NotImplementedError


class StatesResult(PklResult):
    def __init__(self, states):
        """list of state objects"""
        super().__init__(states)

    @classmethod
    def FILENAME(cls):
        return "states.pkl"

    @classmethod
    def gather(cls, results):
        raise NotImplementedError

    @classmethod
    def save_gathered_results(cls, gathered_results, path):
        raise NotImplementedError


class HistoryResult(PklResult):
    def __init__(self, history):
        """list of state objects"""
        super().__init__(history)

    @classmethod
    def FILENAME(cls):
        return "history.pkl"
