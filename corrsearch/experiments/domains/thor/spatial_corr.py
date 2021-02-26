from corrsearch.utils import indicator, euclidean_dist

def uniform(*args):
    return True

def nearby(point1, point2, radius=2):
    return euclidean_dist(point1, point2) <= radius

def not_nearby(point1, point2, radius=2):
    return not nearby(point1, point2, radius=radius)
