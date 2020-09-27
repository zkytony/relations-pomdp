############## Actual maps; Returning GridMap objects ##########

from relpomdp.home2d.domain.maps.grid_map import GridMap
from relpomdp.home2d.domain.maps.build_map import *

def map_small_0(seed=100):
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

