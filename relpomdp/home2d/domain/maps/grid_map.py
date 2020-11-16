import pomdp_py
import copy
from relpomdp.home2d.domain.relation import *
from relpomdp.home2d.domain.state import *
from relpomdp.home2d.domain.action import *
from relpomdp.home2d.domain.condition_effect import *
from relpomdp.oopomdp.framework import Objstate
from relpomdp.utils_geometry import intersect, euclidean_dist
import math
import json
import sys

class Room:
    def __init__(self, name, walls, locations, doorways=None):
        """walls: A set of (x,y,"H"|"V") walls, (not WallState!)
        locations: A set of (x,y) locations.
        name (str): Assumed to be of the format Class-#
        doorways (set): A set of (x,y) locations that are at the doorway.
            Default is empty."""
        self.name = name
        self.walls = walls
        self.locations = locations
        self.room_type = self.name.split("-")[0]
        if doorways is None:
            doorways = set()  # must not use set() as default parameter
        self.doorways = doorways

        # Compute top-left corner and width/length; The room is,
        # however, not always a rectangle.
        self.top_left = (
            min(locations, key=lambda l: l[0])[0],
            min(locations, key=lambda l: l[1])[1]
        )
        self.width = max(locations, key=lambda l: l[0])[0] - self.top_left[0] + 1
        self.length = max(locations, key=lambda l: l[1])[1] - self.top_left[1] + 1

        mean = np.mean(np.array([*self.locations]),axis=0)
        self._center_of_mass = tuple(np.round(mean).astype(int))

    # def to_state(self):
    #     return ContainerState(self.room_type, self.name, tuple(self.locations))

    @property
    def center_of_mass(self):
        return self._center_of_mass

    def __str__(self):
        return "Room(%s;%d,%d,%d,%d)" % (self.name, self.top_left[0], self.top_left[1],
                                         self.width, self.length)

    def __repr__(self):
        return str(self)

    def add_doorway_by_wall(self, wall):
        """Adds the x,y grid cell that touches the wall
        as the doorway; You can think of this as, the wall
        is removed, and the x,y grid cell is the 'entrance'
        to the room.

        The wall is represented as a tuple (x, y, direction)
        """
        wx, wy, direction = wall
        if (wx, wy) in self.locations:
            self.doorways.add((wx, wy))
            return

        if direction == "V":
            # Vertical wall. So either the doorway is at wx, wy,
            # or it is at wx+1, wy, whichever is a valid location in
            # this room. (wx, wy) case has been checked above
            assert (wx+1, wy) in self.locations,\
                "Expecting room location on right side of vertical wall."
            self.doorways.add((wx+1, wy))
        else:
            # Horizontal wall. Similar reasoning
            assert (wx, wy+1) in self.locations,\
                "Expecting room location on above the horizontal wall at (%d, %d)."\
                % (wx, wy+1)
            self.doorways.add((wx, wy + 1))



class GridMap:
    def __init__(self, width, length, walls, rooms, name="grid_map"):
        """
        walls (dict): Map from objid to WallState.
        rooms (list): A list of rooms
            (includes both rooms and corridors)"""
        self.width = width
        self.length = length
        self.walls = walls
        self.rooms = {r.name:r for r in rooms}

        # Create a mapping from location to room
        self.xy_to_room = {}
        for name in self.rooms:
            room = self.rooms[name]
            for x,y in room.locations:
                self.xy_to_room[(x,y)] = room.name

        # all locations are free
        self.free_locations = {(x,y) for x in range(width)
                               for y in range(length)}

        self._name = name

    @property
    def name(self):
        return self._name

    def room_of(self, position):
        if position in self.xy_to_room:
            return self.rooms[self.xy_to_room[position]]
        else:
            return None

    def same_room(self, loc1, loc2):
        """Returns true if loc1 and loc2 (both x,y) are in the same room.
        Returns false if the two rooms are different, or if one of the
        locations does not have a room."""
        room1 = self.room_of(loc1)
        room2 = self.room_of(loc2)
        if room1 is not None and room2 is not None:
            return room1 == room2
        else:
            return False

    def within_bounds(self, position):
        if not (position[0] >= 0 and position[0] < self.width\
                and position[1] >= 0 and position[1] < self.length):
            return False
        return True

    def containers(self, container_type):
        if container_type == "Room":
            return {name:self.rooms[name].to_state()
                    for name in self.rooms}

    def legal_motions_at(self, x, y, all_motion_actions, permitted_locations=None):
        """
        permitted_locations (set): A set of (x,y) locations besides what is in
            that we allow the robot to move to. If None, `self.free_locations`
            will be used.
        """
        so = Objstate("PoseObject", pose=(x,y))
        motion_actions = set(all_motion_actions)
        legal_actions = set()
        if permitted_locations is None:
            permitted_locations = self.free_locations

        # Compute maximum expected length of travel
        max_traj_len = float("-inf")
        for a in motion_actions:
            dx, dy, dth = a.motion
            traj = (x,y), (x+dx, y+dy)
            max_traj_len = max(max_traj_len, math.sqrt(dx**2 + dy**2))

        # Find walls that are nearby
        nearby_walls = set()
        for wall_id in self.walls:
            p1, p2 = self.walls[wall_id].segment
            if euclidean_dist(p1, (x,y)) <= max_traj_len\
               or euclidean_dist(p2, (x,y)) <= max_traj_len:
                nearby_walls.add(wall_id)

        # Compute legal motions
        for a in motion_actions:
            dx, dy, dth = a.motion
            if x+dx < 0:
                continue
            if y+dy < 0:
                continue
            if (x+dx, y+dy) not in permitted_locations:
                continue
            legal = True
            for wall_id in nearby_walls:
                if self.walls[wall_id].intersect((x,y), (x+dx, y+dy)):
                    legal = False
                    break
            if legal:
                legal_actions.add(a)
        return legal_actions


    def compute_legal_motions(self, all_motion_actions):
        """Returns a map from (x,y) to legal motion actions"""
        legal_actions = {}  #
        for x in range(self.width):
            for y in range(self.length):
                legal_actions[(x,y)] = self.legal_motions_at(x, y, all_motion_actions)
        return legal_actions
