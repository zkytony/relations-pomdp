from sciex import *
import numpy as np
from scipy import stats
import math
import os
import json
import yaml
import copy
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)


class HistoryResult(PklResult):
    def __init__(self, history):
        """list of state objects"""
        super().__init__(history)

    @classmethod
    def FILENAME(cls):
        return "history.pkl"

    @classmethod
    def gather(cls, results):
        """`results` is a mapping from specific_name to a dictionary {seed: actual_result}.
        Returns a more understandable interpretation of these results"""
        # Returns the number of objects detected at the end.
        pass
    
    @classmethod
    def save_gathered_results(cls, gathered_results, path):
        pass
