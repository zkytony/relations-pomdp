import pomdp_py
import copy
import math
import json
import sys
from collections import deque
from corrsearch.objects import *
from corrsearch.utils import remap, to_degrees

def neighbors(x,y):
    return [(x+1, y), (x-1,y),
            (x,y+1), (x,y-1)]

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

    def to_thor_pose(self, x, y, th, grid_size=None):
        return (*self.to_thor_pos(x, y, grid_size=grid_size), to_degrees(th))

    def to_thor_pos(self, x, y, grid_size=None):
        """
        Given a point (x,y) in the grid map, convert it to (x,z) in
        the THOR coordinte system (grid size is accounted for).
        If grid_size is None, will return the integers
        for the corresponding coordinate.
        """
        # Note that y is z in Unity
        thor_gx_min, thor_gx_max = self.ranges_in_thor[0]
        thor_gy_min, thor_gy_max = self.ranges_in_thor[1]
        thor_gx = remap(x, 0, self.width, thor_gx_min, thor_gx_max)
        thor_gy = remap(y, 0, self.length, thor_gy_min, thor_gy_max)
        if grid_size is not None:
            # Snap to grid
            return (grid_size * round((thor_gx * grid_size) / grid_size),
                    grid_size * round((thor_gy * grid_size) / grid_size))
        else:
            return (thor_gx, thor_gy)

    def to_grid_pos(self, thor_x, thor_z, grid_size=None):
        """
        Convert thor location to grid map location. If grid_size is specified,
        then will regard thor_x, thor_z as the original Unity coordinates.
        If not, then will regard them as grid indices but with origin not at (0,0).
        """
        if grid_size is not None:
            thor_gx = thor_x // grid_size
            thor_gy = thor_z // grid_size
        else:
            thor_gx = thor_x
            thor_gy = thor_z

        # remap coordinates to be nonnegative (origin AT (0,0))
        thor_gx_min, thor_gx_max = self.ranges_in_thor[0]
        thor_gy_min, thor_gy_max = self.ranges_in_thor[1]
        gx = int(remap(thor_gx, thor_gx_min, thor_gx_max, 0, self.width))
        gy = int(remap(thor_gy, thor_gy_min, thor_gy_max, 0, self.length))
        return gx, gy


    def free_region(self, x, y):
        """Given (x,y) location, return a set of locations
        that are free and connected to (x,y)"""
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

    def boundary_cells(self, thickness=1):
        """
        Returns a set of locations corresponding to
        obstacles that lie between free space and occluded spaces.
        These are usually locations where objects are placed.
        """
        last_boundary = set()
        for i in range(thickness):
            boundary = set()
            for x, y in self.obstacles:
                for nx, ny in neighbors(x, y):
                    if (nx, ny) in self.free_locations\
                       or (nx, ny) in last_boundary:
                        boundary.add((x,y))
                        break
            last_boundary.update(boundary)
        return last_boundary
