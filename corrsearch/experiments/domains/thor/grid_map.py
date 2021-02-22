import pomdp_py
import copy
import math
import json
import sys
from collections import deque
from corrsearch.objects import *
from corrsearch.utils import remap

class GridMap:
    """
    A grid map is a collection of locations, walls, and some locations have obstacles.
    The coordinate convention differs from THOR. Horizontal axis is x, vertical axis is y.

    Note that the coordinates in grid map starts from (0,0) to (width-1,length-1).
    This differs from THOR which could have negative coordinates
    """

    def __init__(self, width, length, obstacles, name="grid_map",
                 ranges_in_thor=None):
        """
        xpos (list): list of x coordinates for free cells
        ypos (list): list of y coordinates for free cells
        obstacles (set): a set of locations for the obstacles
        ranges_in_thor (tuple): A tuple (thor_gx_range, thor_gy_range)
            where thor_gx_range are the min max range for grid x coordiates in thor
            where thor_gy_range are the min max range for grid y coordiates in thor
            Note that thor grid coordinates do not originate in (0,0) but at some other
                point determined by Unity.
        """
        self.width = width
        self.length = length

        self.obstacles = obstacles
        self.free_locations = {(x,y) for x in range(width)
                               for y in range(length)
                               if (x,y) not in obstacles}

        self.name = name
        self.ranges_in_thor = ranges_in_thor

    def to_thor_pos(self, x, y, grid_size=None):
        """
        Given a point (x,y) in the grid map, convert it to (x,z) in
        the THOR coordinte system (grid size is accounted for).
        If grid_size is None, will return the integers
        for the corresponding coordinate.
        """
        # Note that y is z in Unity
        thor_gx_range = self.ranges_in_thor[0]
        thor_gy_range = self.ranges_in_thor[1]
        thor_gx = remap(x, 0, self.width, thor_gx_range[0], thor_gx_range[1])
        thor_gy = remap(y, 0, self.width, thor_gy_range[0], thor_gy_range[1])
        if grid_size is not None:
            return (thor_gx * grid_size, thor_gy * grid_size)
        else:
            return (thor_gx, thor_gy)

    def free_region(self, x, y):
        """Given (x,y) location, return a set of locations
        that are free and connected to (x,y)"""
        def neighbors(x,y):
            return [(x+1, y), (x-1,y),
                    (x,y+1), (x,y-1)]
        region = set()
        q = deque()
        q.append((x,y))
        visited = set()
        while len(q) > 0:
            loc = q.popleft()
            region.add(loc)
            for nb_loc in neighbors(*loc):
                if nb_loc in self.free_locations:
                    if nb_loc not in visited:
                        visited.add(nb_loc)
                        q.append(nb_loc)
        return region
