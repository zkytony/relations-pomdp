############## Actual maps; Returning GridMap objects ##########

from relpomdp.home2d.domain.maps.grid_map import GridMap
from relpomdp.home2d.domain.maps.build_map import *

def map_small_1(seed=100):
    walls = init_map(10,10)
    room1 = make_room("Bathroom-1", 0,7,3,3)
    room2 = make_room("Kitchen-2", 0,0,3,7)
    room3 = make_room("Office-3", 5,7,5,3)
    room4 = make_room("Office-4", 5,0,5,4)
    rooms = [room1, room2, room3, room4]
    corridor1, rooms = make_corridor("Corridor-1", 3, 0, 2, 10, rooms, seed=seed)
    corridor2, rooms, corridors = make_corridor("Corridor-2", 5, 4, 5, 3, rooms, [corridor1], seed=seed)
    corridors.append(corridor2)

    for room in rooms:
        walls |= set(room.walls)
    for cr in corridors:
        walls |= set(cr.walls)
    wall_states = walls_to_states(walls)
    return GridMap(10, 10, wall_states, rooms + corridors)
