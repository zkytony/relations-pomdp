import pomdp_py
from pomdp_py.algorithms.value_function import *
import numpy as np
import copy

def valueM(b, M, horizon=1):
    return value(b, M["S"], M["A"], M["Z"], M["T"],
                 M["O"], M["R"], M["gamma"], horizon=horizon)

def extended_valueM(b, M, horizon=1):
    return extended_value(b, M["S"], M["A"], M["Z"],
                          M["T"], M["O"], M["R"], M["gamma"], horizon=horizon)

def extended_value(b, S, A, Z, T, O, R, gamma, horizon=1):
    """
    Computes the value of a POMDP at belief state b,
    given a POMDP defined by S, A, Z, T, O, R and gamma.

    Args:
        b (dict or Histogram): belief state, maps from every state in S to a probability
        T (TransitionModel): The pomdp_py.TransitionModel where probability is defined
        O (ObservationModel): The pomdp_py.ObservationModel where probability is defined
        R (RewardModel): The pomdp_py.RewardModel: deterministic
        gamma (float): The discount factor
        horizon (int): The planning horizon (rewards are accumulated up
                       to the planning horizon).
    Returns:
        float: value at belief
    """
    b = copy.deepcopy(b)
    if type(b) == list or isinstance(b, np.ndarray):
        b = {S[i]:b[i] for i in range(len(S))}

    total_prob = sum(b[s] for s in b)
    for s in b:
        # normalize
        b[s] /= total_prob
    return value(b, S, A, Z, T, O, R, gamma, horizon=horizon)
