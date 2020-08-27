from relpomdp.object_search.world_specs.build_world import *
from relpomdp.object_search.env import *

if __name__ == "__main__":
    # Build a world
    walls = init_world(10,10)
    room1 = make_room(5, 5, 3, 3)
    room2 = make_room(0, 2, 3, 3)
    corridor, rooms = make_corridor(0, 0, 6, 2, [room1, room2])
    corridor, rooms, corridors = make_corridor(6, 0, 2, 5, [room1, room2], [corridor])
    for room in rooms:
        walls |= room
    for cr in corridors:
        walls |= cr
    walls |= corridor
    wall_states = walls_to_states(walls)
    grid_map = GridMap(10, 10, wall_states)

    # Arbitrary states
    robot_state = RobotState((0,0), "+x")
    salt_state = ItemState("Salt", (3,3))
    pepper_state = ItemState("Pepper", (3,2))
    init_state = {1: robot_state,
                  10: salt_state,
                  15: pepper_state}
    env = ObjectSearchEnvironment(grid_map,
                                  init_state,
                                  {10})
    viz = ObjectSearchViz(env,
                          {10: (128, 128, 128),
                           15: (200, 10, 10)},
                          res=40,
                          controllable=True,
                          img_path="../imgs")
    viz.on_init()
    viz.on_execute()
