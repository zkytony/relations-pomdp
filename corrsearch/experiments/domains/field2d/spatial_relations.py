from corrsearch.utils import euclidean_dist

def uniform(*args):
    return True

def nearby(point1, point2, radius=2):
    return euclidean_dist(point1, point2) <= radius
