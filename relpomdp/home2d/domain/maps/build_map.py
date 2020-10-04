import numpy as np
from relpomdp.home2d.domain.state import WallState
from relpomdp.home2d.domain.maps.grid_map import GridMap
import random
import pickle

class Room:
    def __init__(self, name, walls, locations):
        """walls: A set of (x,y,"H"|"V") walls,
        locations: A set of (x,y) locations.
        name (str): Assumed to be of the format Class-#"""
        self.name = name
        self.walls = walls
        self.locations = locations
        self.room_type = self.name.split("-")[0]

        mean = np.mean(np.array([*self.locations]),axis=0)
        self._center_of_mass = tuple(np.round(mean).astype(int))
        
    # def to_state(self):
    #     return ContainerState(self.room_type, self.name, tuple(self.locations))

    @property
    def center_of_mass(self):
        return self._center_of_mass
    
def init_map(width, length):
    """
    Create a map without any inner wall and only
    walls around the map. Note that for a horizontal
    wall at (x,y), it is on the north edge of that cell.
    A vertical wall is on the east side of the grid cell.
    ---
    A map is represented by a collection of walls (x,y,"H"|"V").
    """
    top_walls = [(x,length-1,"H") for x in range(width)]
    bottom_walls = [(x,-1,"H") for x in range(width)]
    left_walls = [(-1,y,"V") for y in range(length)]
    right_walls = [(width-1,y,"V") for y in range(length)]
    return set(top_walls + bottom_walls + left_walls + right_walls)

def make_room(name, x, y, width, length):
    """
    makes a room, which has bottom-left corner at x,y and with
    dimensions width and length.
    """
    walls = init_map(width, length)
    # shift all the walls
    res = []
    for wx, wy, direction in walls:
        res.append((wx+x, wy+y, direction))
    # Get the locations -- it's just the rectangle box
    locations = {(px,py)
                  for px in range(x,x+width)
                  for py in range(y,y+length)}
    return Room(name, set(res), locations)

def make_corridor(name, x, y, width, length, rooms, other_corridors=[], seed=100):
    """
    Adds a corridor, which is also a rectangle with bottom-left
    coordinates (x,y) and dimensions (width, length).

    Except that for every room (each room is a set of walls),
    one of the walls that overlap with the corridor wall
    will be removed as doorway.

    Also that all walls that the new corridor has that intersect
    with walls in existing `corridors` will be removed, effectively
    connecting corridors.

    Returns corridor, rooms, corridors; The first is the walls for the
    corridor, and the second is a list of rooms each a set of walls.

    The seed determines the order of walls which will make an impact on
    the doorway
    """
    random.seed(seed)
    rooms = list(rooms)
    corridor = make_room(name, x, y, width, length)
    for room in rooms:
        walls = list(sorted(room.walls))
        random.shuffle(walls)
        for wall in walls:
            if wall in corridor.walls:
                room.walls.remove(wall)
                corridor.walls.remove(wall)
                break

    other_corridors = list(other_corridors)
    for other_corridor in other_corridors:
        walls = list(sorted(other_corridor.walls))
        random.shuffle(walls)        
        for wall in walls:
            if wall in corridor.walls:
                other_corridor.walls.remove(wall)
                corridor.walls.remove(wall)

    if len(other_corridors) > 0:
        return corridor, rooms, other_corridors
    else:
        return corridor, rooms
    
def walls_to_states(walls, base_id=1000):
    # Remove duplicated walls and convert the unique ones into states
    wall_states = {}
    for i, tup in enumerate(sorted(set(walls))):
        x, y, direction = tup
        wall_states[base_id+i] = WallState((x,y), direction)
    return wall_states

