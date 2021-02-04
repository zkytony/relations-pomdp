import math

def indicator(cond):
    return 1.0 if cond else 0.0

def euclidean_dist(p1, p2):
    return math.sqrt(sum([(a - b)** 2 for a, b in zip(p1, p2)]))
