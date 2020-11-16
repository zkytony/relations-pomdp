from relpomdp.home2d.domain.maps.grid_map import GridMap
import numpy as np

class PartialGridMap(GridMap):

    def __init__(self, free_locations, walls, name="partial_grid_map"):
        """
        free_locations (set): a set of (x,y) locations that are free
        walls (dict): map from objid to WallState
        """
        self.free_locations = free_locations
        self.walls = walls
        self._cell_to_walls = self._compute_walls_per_cell()
        self._room_locations = {}  # maps from room name to a set of locations in the room
        self._location_to_room = {}  # maps from location to room name

        width, length = self._compute_dims(free_locations)
        super().__init__(width, length, walls, {}, name=name)

    def _compute_dims(self, free_locs):
        width = (max(point[0] for point in free_locs) - min(point[0] for point in free_locs)) + 1
        length = (max(point[1] for point in free_locs) - min(point[1] for point in free_locs)) + 1
        return width, length

    def _compute_walls_per_cell(self):
        """
        Returns a map that maps from each location to a set of walls (wall ids)
        that surrounds this cell
        """
        cell_to_walls = {}
        for wall_id in self.walls:
            cell1, cell2 = self.walls[wall_id].cells_touching()
            if cell1 not in cell_to_walls:
                cell_to_walls[cell1] = set()
            if cell2 not in cell_to_walls:
                cell_to_walls[cell2] = set()
            cell_to_walls[cell1].add(wall_id)
            cell_to_walls[cell2].add(wall_id)
        return cell_to_walls

    def room_of(self, position):
        if position in self._location_to_room:
            return self._location_to_room[position]
        return None

    def same_room(self, loc1, loc2):
        """Returns true if loc1 and loc2 (both x,y) are in the same room"""
        try:
            if self._location_to_room[loc1] is not None\
               and self._location_to_room[loc2] is not None:
                return self._location_to_room[loc1] == self._location_to_room[loc2]
            else:
                return False
        except KeyError:
            # KeyError could occur
            return False


    def update(self, free_locs, walls, loc_to_room={}):
        self.free_locations |= free_locs
        self.walls.update(walls)
        self.width, self.length = self._compute_dims(self.free_locations)
        self._cell_to_walls = self._compute_walls_per_cell()

        # Update room tracking
        for loc in loc_to_room:
            room = loc_to_room[loc]
            if room not in self._room_locations:
                self._room_locations[room] = set()
            self._room_locations[room].add(loc)
            if loc in self._location_to_room:
                assert self._location_to_room[loc] == room
            self._location_to_room[loc] = room


    def frontier(self):
        """Returns a set of locations that is an immediate
        expansion of locations at the edge of the current map"""
        frontier = set()
        for x, y in self.free_locations:
            # Check all four directions of this grid cell and
            # see if there is one side that extends into the unknown
            connecting_cells = {(x+1, y), (x-1,y), (x,y+1), (x,y-1)}
            if (x,y) in self._cell_to_walls:
                surrounding_walls = self._cell_to_walls[(x,y)]
                for wall_id in surrounding_walls:
                    wall = self.walls[wall_id]
                    blocked_loc = set(wall.cells_touching()) - set({(x,y)})
                    connecting_cells -= blocked_loc

            for cell in connecting_cells:
                if cell not in self.free_locations\
                   and (cell[0] >= 0 and cell[1] >= 0):
                    # This is a frontier, because it is not blocked by
                    # any wall and is not in a free location, and it
                    # does not have negative coordinate s
                    frontier.add((x,y))
        return frontier

    def compute_legal_motions(self, all_motion_actions):
        """This is done by creating a map from
        current free locations and frontier to a set of
        motions that can be executed there."""
        legal_actions = {}
        all_locations = self.free_locations | self.frontier()
        for x, y in all_locations:
            legal_actions[(x,y)] = self.legal_motions_at(x, y, all_motion_actions,
                                                         permitted_locations=all_locations)
        return legal_actions
