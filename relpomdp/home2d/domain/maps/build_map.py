import numpy as np
from relpomdp.home2d.domain.state import WallState
from relpomdp.home2d.domain.maps.grid_map import GridMap
from relpomdp.oopomdp.framework import Objstate
import random
import pickle

class Room:
    def __init__(self, name, walls, locations, doorways=None):
        """walls: A set of (x,y,"H"|"V") walls,
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
    Here, `room_tup` is a tuple (top_left, width, length)

    Helper for _generate_rooms """
    top_left, width, length = room_tup
    for rm in rooms:
        for dx in range(width):
            for dy in range(length):
                if (top_left[0] + dx, top_left[1] + dy) in rm.locations:
                    return True
    return False


def pcg_map(width, length, nrooms, categories, seed=None,
            min_room_size=2, max_room_size=6, max_trys=100, ndoors=1):
    """
    Procedurally generates a map (with no objects).

    The (0,0) coordinates is at top left of the map.

    Args:
        width (int): width of the map
        length (int): length of the map
        nrooms (int): Number of rooms  (each room is a rectangle)
        categories (list): Category of each room. Length must be at least `nrooms`.
        seed (int): Random seed
        min_room_size (int): Minimum room width/length
        max_room_size (int): Maximum room width/length
        max_trys (int): Maximum number of attempts to insert valid rooms into free space.
        ndoors (int): Number of walls to be removed (connected to the corridor) to
            give space for a doorway.
    Returns:
        GridMap
    """
    if seed is not None:
        random.seed(seed)
    border_walls = init_map(width, length)

    free_locations = {(x,y)
                      for x in range(width)
                      for y in range(length)}
    bad_candidates = set()
    rooms = []
    trys = 0

    # First, insert valid boxes as rooms
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

    # Then, add doorway
    # Iterate over walls; If a wall is not overlapping with any other wall,
    # then we could remove it to make a doorway. We can add a certain number
    # of doors per room.
    ## First, create a mapping from wall to rooms
    name_to_rooms = {rm.name:rm for rm in rooms}
    wall_to_rooms = {}
    for rm in rooms:
        for wall in rm.walls:
            if wall in border_walls:
                continue # skip border walls
            if wall not in wall_to_rooms:
                wall_to_rooms[wall] = []
            wall_to_rooms[wall].append(rm.name)
    ## Then, for all walls which is only shared by one room, add
    ## this wall to be potentially removed, if the total number of
    ## walls to be removed for the room is below the `ndoors` limit.
    room_doors = {}
    for wall in wall_to_rooms:
        rooms_sharing_wall = wall_to_rooms[wall]
        if len(rooms_sharing_wall) == 1:
            rm_name = rooms_sharing_wall[0]
            if rm_name not in room_doors:
                room_doors[rm_name] = set()
            if len(room_doors[rm_name]) < ndoors:
                room_doors[rm_name].add(wall)
    ## Remove candidate walls identified above.
    rm1, rm2 = list(room_doors.keys())[:2]
    print(name_to_rooms[rm1].doorways is name_to_rooms[rm2].doorways)
    for rm_name in room_doors:
        doorway_walls = room_doors[rm_name]
        # Remove these walls from the room
        name_to_rooms[rm_name].walls -= doorway_walls
        # Add the locations at the doorway to the room
        # print("YOYO", doorway_walls)
        for wall in doorway_walls:
            name_to_rooms[rm_name].add_doorway_by_wall(wall)

    # Gather all walls; This is needed by the GridMap creation.
    all_walls = border_walls
    for room in rooms:
        all_walls |= set(room.walls)

    # Everything else is a corridor which has no wall.
    rooms.append(Room("Corridor-%d" % len(rooms), set(), free_locations))

    # Verify there is no overlap
    occupied = set()
    for room in rooms:
        for loc in room.locations:
            if loc not in occupied:
                occupied.add(loc)
            else:
                raise ValueError("There is clearly overlap.")

    wall_states = walls_to_states(all_walls)
    gm = GridMap(width, length, wall_states, rooms)
    print(gm.rooms[list(gm.rooms)[0]].doorways)
    return gm


def _placeable(obj_tup, free_locations):
    top_left, width, length = obj_tup
    # Check if all locations within this box are free
    for x in range(width):
        for y in range(length):
            loc = (top_left[0] + x,
                   top_left[1] + y)
            if loc not in free_locations:
                return False
    return True

def pcg_world(grid_map, objects, max_trys=30):
    """
    Procedurally generates a random world, given a grid map and a specification
    of objects per room category.

    Args:
        objects (dict): Specifies what objects would appear within a room category
            Format: {room_category -> {object_class -> (amount, footprint)}}
              footprint is essentially the (width,length) tuple of the bounding box

    Returns:
        init_state (dict): Mapping from object id to object state
    """
    init_state = {}
    for i, room_name in enumerate(grid_map.rooms):
        room = grid_map.rooms[room_name]
        if room.room_type not in objects:
            print("No object will get generated in %s" % room.room_type)
            continue
        spec = objects[room.room_type]

        # Create a distribution such that the boundary of the room is
        # more likely for objects to appear...
        free_locations = set(room.locations)

        # Try to generate according to the spec as much as possible
        for j, objclass in enumerate(spec):
            amount, footprint = spec[objclass]
            new_objects = {}
            trys = 0
            bad_candidates = set()
            while len(new_objects) < amount:
                skip = False

                top_left = random.sample(free_locations, 1)[0]
                width, length = footprint
                obj_tup = (top_left, width, length)
                if obj_tup in bad_candidates:
                    skip = True
                elif not _placeable(obj_tup, free_locations):
                    skip = True

                if skip:
                    bad_candidates.add(obj_tup)
                else:
                    objid = ((i+1)*1000 + (j+1)*100) + len(new_objects)
                    if width == 1 and length == 1:
                        object_state = Objstate(objclass,
                                                pose=top_left)
                    else:
                        object_state = Objstate(objclass,
                                                p=top_left,
                                                w=width,
                                                l=length)
                    new_objects[objid] = object_state
                    free_locations -= {(top_left[0]+x,
                                        top_left[1]+y) for x in range(width) for y in range(length)}

                trys += 1
                if trys > max_trys:
                    print("Unable to generate all instances of object class %s"\
                          "(likely not enough space)" % objclass)
                    break
            init_state.update(new_objects)
    return init_state


def random_world(width, length, nrooms, categories, objects={},
                 min_room_size=2, max_room_size=6, seed=None,
                 ndoors=2, robot_id=0, init_robot_pose=(0, 0, 0)):
    grid_map = pcg_map(width, length, nrooms, categories,
                       min_room_size=min_room_size,
                       max_room_size=max_room_size,
                       seed=seed, ndoors=ndoors)
    init_state = pcg_world(grid_map, objects)
    init_state[robot_id] = Objstate("Robot",
                                    pose=init_robot_pose,
                                    camera_direction="-x")
    return init_state, grid_map
