"""
Conversion between POMDP and THOR
"""
import numpy as np
from corrsearch.experiments.domains.thor.grid_map import *
from corrsearch.experiments.domains.thor.thor import *

def convert_scene_to_grid_map(controller, grid_size):
    x, z = reachable_thor_pos2d(controller)

    # obtain grid indices for coordinates
    gx = (x // grid_size).astype(int)
    gz = (z // grid_size).astype(int)

    # grid map positions
    positions = set(zip(gx, gz))

    # grid map dimensions
    width = max(gx) - min(gx)
    length = max(gz) - min(gz)

    grid_map = GridMap(width, length, {}, set({(1,0)}))

    return grid_map
