import numpy as np
from relpomdp.object_search.state import WallState
from relpomdp.object_search.grid_map import GridMap
import random

def init_world(width, length):
    """
    Create a world without any inner wall and only
    walls around the world. Note that for a horizontal
    wall at (x,y), it is on the north edge of that cell.
    A vertical wall is on the east side of the grid cell.
    ---
    A world is represented by a collection of walls (x,y,"H"|"V").
    """
    top_walls = [(x,length-1,"H") for x in range(width)]
    bottom_walls = [(x,-1,"H") for x in range(width)]
    left_walls = [(-1,y,"V") for y in range(length)]
    right_walls = [(width-1,y,"V") for y in range(length)]
    return set(top_walls + bottom_walls + left_walls + right_walls)


def make_room(x, y, width, length):
    """
    makes a room, which has bottom-left corner at x,y and with
    dimensions width and length.
    """
    walls = init_world(width, length)
    # shift all the walls
    res = []
    for wx, wy, direction in walls:
        res.append((wx+x, wy+y, direction))
    return set(res)

def make_corridor(x, y, width, length, rooms, corridors=[], seed=100):
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
    corridor = make_room(x, y, width, length)
    for room in rooms:
        walls = list(sorted(room))
        random.shuffle(walls)
        for wall in walls:
            if wall in corridor:
                room.remove(wall)
                corridor.remove(wall)
                break

    corridors = list(corridors)
    for cr in corridors:
        walls = list(sorted(cr))
        random.shuffle(walls)        
        for wall in walls:
            if wall in corridor:
                cr.remove(wall)
                corridor.remove(wall)

    if len(corridors) > 0:
        return corridor, rooms, corridors
    else:
        return corridor, rooms
    
def walls_to_states(walls, base_id=1000):
    # Remove duplicated walls and convert the unique ones into states
    wall_states = {}
    for i, tup in enumerate(sorted(set(walls))):
        x, y, direction = tup
        wall_states[base_id+i] = WallState((x,y), direction)
    return wall_states


############## Actual worlds; Returning GridMap objects ##########
def small_world1(seed=100):
    walls = init_world(10,10)
    room1 = make_room(0,7,3,3)
    room2 = make_room(0,4,3,3)
    room3 = make_room(0,0,3,4)
    room4 = make_room(5,7,2,3)
    room5 = make_room(7,7,3,3)
    room6 = make_room(5,0,2,4)
    room7 = make_room(7,0,3,4)
    rooms = [room1, room2, room3, room4, room5, room6, room7]
    corridor1, rooms = make_corridor(3, 0, 2, 10, rooms, seed=seed)
    corridor2, rooms, corridors = make_corridor(5, 4, 5, 3, rooms, [corridor1], seed=seed)
    corridors.append(corridor2)

    for room in rooms:
        walls |= set(room)
    for cr in corridors:
        walls |= set(cr)
    wall_states = walls_to_states(walls)
    return GridMap(10, 10, wall_states)
        
    
    

