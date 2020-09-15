import relpomdp.oopomdp.framework as oopomdp

"""
Each relation is associated with a probability
function that is used to do belief update.
"""

class Near(oopomdp.InfoRelation):
    def __init__(self, class1, class2, negate=False):
        """
        Args:
            grid_map: Having grid map implies assuming knowing at least the room layout
        """
        self.negate = negate
        self.grid_map = None
        super().__init__("near", class1, class2)

    def probability(self, observation1, observation2):
        is_near = self.is_near(observation1.pose, observation2.pose)
        if not negate:
            return 1.0 - 1e-9 if is_near else 1e-9
        else:
            return 1e-9 if is_near else 1.0 - 1e-9

    def ground(self, grid_map):
        self.grid_map = grid_map

    @property
    def grounded(self):
        return self.grid_map is not None

class GroundLevelNear(Near):
    def is_near(self, pose1, pose2):
        same_room = (self.grid_map.room_of(pose1)\
                         == self.grid_map.room_of(pose2))
        return same_room\
            and euclidean_dist(pose1, pose2) <= 2
    def values(self, class_name):
        # Return positions
        return [(x,y)
                for x in range(self.grid_map.width)
                for y in range(self.grid_map.height)]

class RoomLevelNear(Near):
    def is_near(self, pose1, pose2):
        same_room = (self.grid_map.room_of(pose1)\
                         == self.grid_map.room_of(pose2))
        return same_room
    def values(self, class_name):
        # Return rooms
        return [room_name for room_name in self.grid_map.rooms]    

    
class In(oopomdp.InfoRelation):
    def __init__(self, item_class, container_class,
                 negate=False):
        self.grid_map = None
        self.negate = negate
        self.item_class = item_class
        self.container_class = container_class
        super().__init__("in", item_class, container_class)
        
    def probability(self, observation1, observation2):
        is_in = self.is_in(observation1.pose, observation2.pose)
        if not negate:
            return 1.0 - 1e-9 if is_in else 1e-9
        else:
            return 1e-9 if is_in else 1.0 - 1e-9

    def ground(self, grid_map):
        self.grid_map = grid_map

    @property
    def grounded(self):
        return self.grid_map is not None

    
class RoomLevelIn(In):
    def is_in(self, pose1, pose2):
        # Both poses are room names
        return pose1 == pose2

    def values(self, class_name):
        # Return rooms
        return [room_name for room_name in self.grid_map.rooms]

