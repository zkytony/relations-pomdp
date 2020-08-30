import numpy as np
from relpomdp.object_search.state import WallState
from relpomdp.object_search.grid_map import GridMap
import random
import pickle

class Room:
    def __init__(self, name, walls, locations):
        """walls: A set of (x,y,"H"|"V") walls,
        locations: A set of (x,y) locations."""
        self.name = name
        self.walls = walls
        self.locations = locations

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

############## Actual maps; Returning GridMap objects ##########
def small_map0(seed=100):
    walls = init_map(5,5)
    room1 = make_room("room-1", 0,0,3,3)
    rooms = [room1]
    corridor1, rooms = make_corridor("corridor-1", 3, 0, 2, 5, rooms, seed=seed)
    corridor2, rooms, corridors = make_corridor("corridor-2", 0, 3, 3, 2, rooms,
                                                [corridor1], seed=seed)
    corridors.append(corridor2)

    for room in rooms:
        walls |= set(room.walls)
    for cr in corridors:
        walls |= set(cr.walls)
    wall_states = walls_to_states(walls)
    return GridMap(5, 5, wall_states, rooms + corridors)


def small_map1(seed=100):
    walls = init_map(10,10)
    room1 = make_room("room-1", 0,7,3,3)
    room2 = make_room("room-2", 0,0,3,7)
    room3 = make_room("room-3", 5,7,5,3)
    room4 = make_room("room-4", 5,0,5,4)
    rooms = [room1, room2, room3, room4]
    corridor1, rooms = make_corridor("corridor-1", 3, 0, 2, 10, rooms, seed=seed)
    corridor2, rooms, corridors = make_corridor("corridor-2", 5, 4, 5, 3, rooms, [corridor1], seed=seed)
    corridors.append(corridor2)

    for room in rooms:
        walls |= set(room.walls)
    for cr in corridors:
        walls |= set(cr.walls)
    wall_states = walls_to_states(walls)
    return GridMap(10, 10, wall_states, rooms + corridors)


def big_map1(width=50, length=50, seed=100):
    walls = init_map(width, length)

    # layers
    corridor_width = width
    corridor_length = 3    
    room_width = width // 10
    room_length = length // 5 - corridor_length

    rooms = []
    for i in range(length // (corridor_length + room_length)):
        for j in range(width // room_width):
            x = j*room_width
            y = i * (corridor_length + room_length)
            room = make_room("room-%d" % (len(rooms)+1), x, y, room_width, room_length)
            rooms.append(room)

    corridors = []
    for k in range(length // (corridor_length + room_length)):
        x = 0
        y = k * (corridor_length + room_length) + room_length
        corridor, rooms = make_corridor("corridor-%d" % (len(corridors)+1), x, y,
                                        corridor_width, corridor_length, rooms, seed=seed)
        corridors.append(corridor)

    for room in rooms:
        walls |= set(room.walls)
    for cr in corridors:
        walls |= set(cr.walls)
    wall_states = walls_to_states(walls)
    return GridMap(width, length, wall_states, rooms + corridors)
