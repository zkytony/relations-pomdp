from relpomdp.oopomdp.abstraction import AbstractAttribute
from relpomdp.object_search.state import Pose
from relpomdp.object_search.world_specs.build_world import small_map1

class Room(AbstractAttribute):
    """This is an abstract attribute for Pose"""
    def __init__(self, room_name):
        super().__init__("room", room_name)
    @property
    def room_name(self):
        return self.value
    def copy(self):
        return Room(self.value)
    def reverse_image(self, grid_map):
        # Returns a set of Pose attributes
        room = grid_map.rooms[self.room_name]
        return [Pose(loc) for loc in room.locations]
    @classmethod
    def abstractify(self, pose_attribute, grid_map):
        if type(pose_attribute) == tuple:
            room_name = grid_map.room_of(pose_attribute)
        else:  # Pose class
            room_name = grid_map.room_of(pose_attribute.value)
        return Room(room_name)


if __name__ == "__main__":
    grid_map = small_map1()
    room_attr = Room("Kitchen-2")
    print(room_attr.reverse_image(grid_map))
    print(Room.abstractify((3,3), grid_map))
