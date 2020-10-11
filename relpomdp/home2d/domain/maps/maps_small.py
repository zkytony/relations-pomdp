
from relpomdp.home2d.domain.maps.grid_map import GridMap
from relpomdp.home2d.domain.maps.build_map import *
from relpomdp.home2d.domain.maps.map_big_0 import map_big_0

def map_small_0(seed=100):
    """This is tiny. Not worth experimenting on"""
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



def map_small_2(seed=101):
    walls = init_map(10,10)
    room1 = make_room("Kitchen-1", 4, 6, 6, 4)
    room2 = make_room("Office-1", 4, 2, 3, 4)
    room3 = make_room("Office-2", 7, 2, 3, 4)
    rooms = [room1, room2, room3]
    
    corridor1, rooms = make_corridor("Corridor-1", 0, 0, 10, 2, rooms, seed=seed)
    corridor2, rooms, corridors = make_corridor("Corridor-2", 0, 2, 4, 8, rooms, [corridor1], seed=seed)
    corridors.append(corridor2)

    for room in rooms:
        walls |= set(room.walls)
    for cr in corridors:
        walls |= set(cr.walls)
    wall_states = walls_to_states(walls)
    return GridMap(10, 10, wall_states, rooms + corridors)


def map_small_3(seed=100):
    walls = init_map(10,10)
    room1 = make_room("Kitchen-1", 0, 6, 4, 4)
    room2 = make_room("Bedroom-1", 6, 0, 4, 6)
    rooms = [room1, room2]
    
    corridor1, rooms = make_corridor("LivingRoom-1", 0, 0, 6, 6, rooms, seed=seed)
    corridor2, rooms, corridors = make_corridor("Corridor-2", 4, 6, 7, 4, rooms, [corridor1], seed=seed)
    corridors.append(corridor2)

    for room in rooms:
        walls |= set(room.walls)
    for cr in corridors:
        walls |= set(cr.walls)
    wall_states = walls_to_states(walls)
    return GridMap(10, 10, wall_states, rooms + corridors)



def map_small_4(seed=100):
    walls = init_map(10,10)
    room1 = make_room("Kitchen-1", 2, 2, 6, 4)
    room2 = make_room("Bathroom-1", 2, 4, 4, 2)
    room3 = make_room("Bedroom-2", 6, 4, 4, 2)
    rooms = [room1, room2, room3]
    
    # corridor1, rooms = make_corridor("LivingRoom-1", 0, 0, 6, 6, rooms, seed=seed)
    # corridor2, rooms, corridors = make_corridor("Corridor-2", 4, 6, 7, 4, rooms, [corridor1], seed=seed)
    # corridors.append(corridor2)
    corridors = []

    for room in rooms:
        walls |= set(room.walls)
    # for cr in corridors:
    #     walls |= set(cr.walls)
    wall_states = walls_to_states(walls)
    return GridMap(10, 10, wall_states, rooms + corridors)
