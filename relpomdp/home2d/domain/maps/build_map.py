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
    
    
def init_map(width, length, return_parts=False):
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
    if return_parts:
        return top_walls, left_walls, bottom_walls, right_walls
    else:
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


# Level up - Procedural Content Generation of the floor plans as well as objects
def _overlapping(room_tup, rooms):
    """Returns true if room_tup overlaps with rooms;
    Here, `room_tup` is a tuple (top_left, width, length)"""
    top_left, width, length = room_tup
    for rm in rooms:
        for dx in range(width):
            for dy in range(length):
                if (top_left[0] + dx, top_left[1] + dy) in rm.locations:
                    return True
    return False


def pcg_map(width, length, nrooms, categories, objects, seed=100,
            min_room_size=2, max_room_size=6, max_trys=100, ndoors=1):
    """
    Procedurally generates a map.

    The (0,0) coordinates is at top left of the map.

    Args:
        width (int): width of the map
        length (int): length of the map
        nrooms (int): Number of rooms  (each room is a rectangle)
        categories (list): Category of each room. Length must be at least `nrooms`.
        objects (dict): Specifies what objects would appear within a room category
        seed (int): Random seed
        min_room_size (int): Minimum room width/length
        max_room_size (int): Maximum room width/length
        max_trys (int): Maximum number of attempts to insert valid rooms into free space.
        ndoors (int): Number of walls to be removed (connected to the corridor) to
            give space for a doorway.
    """
    random.seed(seed)    
    walls = init_map(width, length)
    
    free_locations = {(x,y)
                      for x in range(width)
                      for y in range(length)}
    bad_candidates = set()
    rooms = []
    trys = 0

    i = 0    
    while len(rooms) < nrooms:
        # Generate n rooms
        skip = False

        top_left = random.sample(free_locations, 1)[0]
        room_width = random.randint(min_room_size, max_room_size)
        room_length = random.randint(min_room_size, max_room_size)
        room_tup = (top_left, room_width, room_length)
        
        if room_tup in bad_candidates:
            skip = True
        if not ((top_left[0] + room_width < width)\
           and (top_left[1] + room_length < length)):
            bad_candidates.add(room_tup)
            skip = True
        
        # Check if this room overlaps with any existing rooms
        if not skip:
            overlaps = _overlapping(room_tup, rooms)
            if overlaps:
                bad_candidates.add(room_tup)
            else:
                room_name = "%s-%d" % (categories[len(rooms)], len(rooms))
                x, y = top_left
                print(room_width, room_length)
                room = make_room(room_name, x, y, room_width, room_length)
                rooms.append(room)
                free_locations = free_locations - room.locations
                
        # Make sure there is no infinite loop
        trys += 1
        if trys > max_trys:
            print("Unable to generate all rooms (likely not enough space)")
            break

    # Iterate over walls; If a wall is not overlapping with any other wall,
    # then we could remove it to make a doorway. We can add a certain number
    # of doors per room.
    # First, create a mapping from wall to rooms
    name_to_rooms = {rm.name:rm for rm in rooms}
    wall_to_rooms = {}
    for rm in rooms:
        for wall in rm.walls:
            if wall not in wall_to_rooms:
                wall_to_rooms[wall] = []
            wall_to_rooms[wall].append(rm.name)

    NDOORS = 2
    room_doors = {}
    for wall in wall_to_rooms:
        rooms_sharing_wall = wall_to_rooms[wall]
        if len(rooms_sharing_wall) == 1:
            rm_name = rooms_sharing_wall[0]
            if rm_name not in room_doors:
                room_doors[rm_name] = set()
            if len(room_doors[rm_name]) < NDOORS:
                room_doors[rm_name].add(wall)
    for rm_name in room_doors:
        doorway_walls = room_doors[rm_name]
        # Remove these walls from the room
        name_to_rooms[rm_name].walls -= doorway_walls
            
    for room in rooms:
        walls |= set(room.walls)

    # Everything else is a corridor; with all walls as its walls
    rooms.append(Room("Corridor-%d" % len(rooms), walls, free_locations))

    # Verify there is no overlap
    occupied = set()
    for room in rooms:
        for loc in room.locations:
            if loc not in occupied:
                occupied.add(loc)
            else:
                raise ValueError("There is clearly overlap.")
            

    wall_states = walls_to_states(walls)
    return GridMap(width, length, wall_states, rooms)    
