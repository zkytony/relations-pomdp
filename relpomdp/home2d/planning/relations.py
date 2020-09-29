from relpomdp.oopomdp.infograph import *
from relpomdp.home2d.utils import euclidean_dist


class GridDomain(Domain):
    def __init__(self, width, length):
        self._locs = {(x,y)
                      for x in range(width)
                      for y in range(length)}
        
    def __contains__(self, value):
        if type(value) != tuple\
           or len(value) != 2:
            raise ValueError("Invalid value format %s for GridDomain" % str(value))
        else:
            return value in self._locs

    def values(self):
        return self._locs

class RoomsDomain(Domain):
    def __init__(self, rooms):
        self._rooms = rooms

    def __contains__(self, value):
        if type(value) != str:
            raise ValueError("Invalid value format %s for RoomsDomain" % str(value))
        else:
            return value in self._rooms
    def values(self):
        return self._rooms        

class PoseAttr(Attribute):
    def __init__(self, _class):
        super().__init__(_class, "pose")

class RoomAttr(Attribute):
    def __init__(self, _class):
        super().__init__(_class, "room_id")

class Near(InfoRelation):
    def __init__(self, attr1, attr2, negate=False):
        self.negate = negate
        negatestr = "not-" if negate else ""
        super().__init__(attr1, attr2, "%snear" % negatestr)

    def ground_on_map(self, grid_map):
        def is_near(pose1, pose2):
            same_room = (grid_map.room_of(pose1)\
                         == grid_map.room_of(pose2))
            near = same_room\
                and euclidean_dist(pose1, pose2) <= 2
            if self.negate:
                return not near
            else:
                return near
            
        gd = GridDomain(grid_map.width, grid_map.length)
        super().ground(gd, gd, is_near)
        

class In(InfoRelation):
    def __init__(self, attr1, attr2, negate=False):
        self.negate = negate
        negatestr = "not-" if negate else ""
        super().__init__(attr1, attr2, "%sin" % negatestr)
                 
    def ground_on_map(self, grid_map):
        def is_in(pose1, pose2):
            if type(pose1) == tuple:
                pose1 = grid_map.room_of(pose1).name
            if type(pose2) == tuple:
                pose2 = grid_map.room_of(pose2).name

            is_in = pose1 == pose2
            if self.negate:
                return not is_in
            else:
                return is_in           

        if isinstance(self.attr1, PoseAttr):
            d1 = GridDomain(grid_map.width, grid_map.length)
        elif isinstance(self.attr1, RoomAttr):
            d1 = RoomDomain(grid_map.rooms)
            
        if isinstance(self.attr2, PoseAttr):
            d2 = GridDomain(grid_map.width, grid_map.length)
        elif isinstance(self.attr2, RoomAttr):
            d2 = RoomDomain(grid_map.rooms)
        super().ground(d1, d2, is_in)
