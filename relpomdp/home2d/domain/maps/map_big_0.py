############## Actual maps; Returning GridMap objects ##########

from relpomdp.home2d.domain.maps.grid_map import GridMap
from relpomdp.home2d.domain.maps.build_map import *


def map_big_0(width=50, length=50, seed=100):
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
