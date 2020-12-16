import math

def euclidean_dist(p1, p2):
    return math.sqrt(sum([(a - b)** 2 for a, b in zip(p1, p2)]))

def in_range_inclusive(v, rang):
    """inclusive in range"""
    return rang[0] <= v <= rang[1]
