from relpomdp.openworld.common import Vec2d, Domain
from relpomdp.openworld.grid2d.maps import MAPS
import random
import os


class MetricMap2d(Domain):
    """
    A metric map is essentially an occupancy grid in 2D,
    where it keeps track of free and occupied locations.
    This is of course a discrete map. (No unknowns)
    """
    def __init__(self, map_name=None, width=0, length=0, obstacle_poses=set()):
        """
        One can construct a map from either a given map name,
        or by directly providing the dimensions and obstacle_poses.
        """
        if map_name is not None:
            self.width, self.length, self._obstacle_poses =\
                self.load_map(map_name)
        else:
            self.width = width
            self.length = length
            self._obstacle_poses = obstacle_poses
        self._free_poses = {(x,y)
                            for x in range(width)
                            for y in range(length)
                            if (x,y) not in self._obstacle_poses}

    def check(self, val):
        """
        Args:
            val: True if val is a 2d tuple and it is not overlapping
                with an obstacle
        """
        return len(val) == 2 and tuple(val) not in self._obstacle_poses

    def sample(self, attr_class):
        """returns a random free pose"""
        return attr_class(random.sample(self._free_poses, 1))

    def load_map(self, map_name):
        if map_name not in MAPS:
            raise ValueError("Unknown map %s" % map_name)
        mapstr = MAPS[map_name]()  # the map is a function and you should call it.
        x, y = 0, 0
        w = 0
        obstacles = set()
        for line in mapstr.split("\n"):
            line = line.strip()
            if len(line) == 0:
                continue            
            for c in line:
                if c == "x":
                    obstacles.add((x,y))
                x += 1
            if w == 0:
                w = len(line)
            else:
                assert len(line) == w, "line in map %s are of different length" % map_name
            y += 1
            x = 0
        return w, y, obstacles  # y is now length
        
