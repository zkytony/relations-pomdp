"""
Conversion between POMDP and THOR
"""
import numpy as np
from corrsearch.experiments.domains.thor.grid_map import *
from corrsearch.experiments.domains.thor.thor import *
from corrsearch.utils import remap

def convert_scene_to_grid_map(controller, scene_name, grid_size):
    x, z = reachable_thor_loc2d(controller)

    # obtain grid indices for coordinates  (origin NOT at (0,0))
    thor_gx = (x // grid_size).astype(int)
    thor_gy = (z // grid_size).astype(int)
    width = max(thor_gx) - min(thor_gx)
    length = max(thor_gy) - min(thor_gy)

    # save these for later use
    thor_gx_range = (min(thor_gx), max(thor_gx))
    thor_gy_range = (min(thor_gy), max(thor_gy))

    # remap coordinates to be nonnegative (origin AT (0,0))
    gx = remap(thor_gx, min(thor_gx), max(thor_gx), 0, width).astype(int)
    gy = remap(thor_gy, min(thor_gy), max(thor_gy), 0, length).astype(int)

    # Little test: can convert back
    try:
        assert all(remap(gx, min(gx), max(gx), thor_gx_range[0], thor_gx_range[1]).astype(int) == thor_gx)
        assert all(remap(gy, min(gy), max(gy), thor_gy_range[0], thor_gy_range[1]).astype(int) == thor_gy)
    except AssertionError as ex:
        print("Unable to remap coordinates")
        raise ex

    # grid map positions
    positions = set(zip(gx, gy))

    # grid map dimensions
    # obstacles: locations that do not fall into valid positions
    obstacles = {(x,y)
                 for x in gx
                 for y in gy
                 if (x,y) not in positions}

    grid_map = GridMap(width, length, obstacles,
                       name=scene_name,
                       ranges_in_thor=(thor_gx_range, thor_gy_range))

    return grid_map
