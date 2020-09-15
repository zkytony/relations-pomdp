from relpomdp.oopomdp.framework import Class
from relpomdp.object_search.state import *
from relpomdp.object_search.observation import *

class Subgoal:
    """Subgoal is like a condition that is True when it is achieved.
    Its achieve depends on s,a and it triggers a state transition."""
    # Status
    IP = "IP"
    SUCCESS = "SUCCESS"
    FAIL = "FAIL"
    
    def __init__(self, name):
        """
        status can be "IP": In Progress; "SUCCESS": achieved; or "FAIL": Failed
        """
        self.name = name
    def valid(self, robot_belief):
        """Is this subgoal valid for evaluation (sort of like initiation set of option)"""
        pass
    def achieve(self, next_state, action):
        pass
    def fail(self, state, action):
        pass
    def effect(self, robot_belief):
        """
        Once a subgoal is achieved, there will be some effects. We consider an effect
        to be observation about classes. For example, "reach kitchen" will produce an
        observation about 'Kitchen' (a class type) which is a RoomObservation. Each
        of this observation is associated with a probability value (unnormalized).
        They can be regarded as ``evidence'' if you assume the subgoal is achieved.

        Returns:
            a mapping {(object_class, observation) -> prob}

            For example, one can have a "ReachKitchen" subgoal, which will likely
            output {(Kitchen, RoomObservation(Kitchen)) -> 1.0 ...}

        Args:
            robot_belief (Generativedistribution): Belief used to determine
                effects on that will happen.
        """
        pass

    def __repr__(self):
        return str(self)
    def __str__(self):
        return "%s(%s)" % (self.__class__.__name__, self.name)

class ItemClass(Class):
    def __init__(self, name):
        super().__init__(name)
        self.grid_map = None
    
    def observation_variations(self):
        """Returns a list of all possible object configurations
        for this class that are basically object observations;
        This may not be feasible in some cases."""        
        variations = []
        for x in range(self.grid_map.width):
            for y in range(self.grid_map.length):
                objobs = ItemObservation(self.name, (x,y))
                variations.append(objobs)
        return variations

    def ground(self, grid_map):
        self.grid_map = grid_map

    @property
    def grounded(self):
        return self.grid_map is not None

class SubgoalClass(Class):
    def __init__(self, name, verb_to_subgoal):
        # verb_to_subgoal, before grounding, is a map
        # from verb (string) to SubgoalClass (class)
        # but afterwards maps from verb to a list of subgoals.
        self.verb_to_subgoal = verb_to_subgoal
        super().__init__(name)
    @property
    def accepted_verbs(self):
        return list(self.verb_to_subgoal.keys())
    def subgoal_for(self, verb):
        return self.verb_to_subgoal[verb]
    @property
    def subgoals(self):
        """Returns a mapping from subgoal name to subgoals"""
        res = {}
        for verb in self.verb_to_subgoal:
            if type(self.verb_to_subgoal[verb]) == list:
                for sg in self.verb_to_subgoal[verb]:
                    res[sg.name] = sg
            else:
                sg = self.verb_to_subgoal[verb]
                res[sg.name] = sg
        return res

class RoomClass(SubgoalClass):

    def __init__(self, room_type):
        # When grounded, this should be a map from verb to subgoal objects
        verb_to_subgoal = {
            "Reach": ReachRoomSubgoal,
            "Search": SearchRoomSubgoal
        }
        self.room_type = room_type
        self._knows_room_types = None
        super().__init__(room_type, verb_to_subgoal)

    @property
    def grounded(self):
        # If grounded, the subgoals should be Subgoal objects or list of subgoal objects
        return isinstance(self.verb_to_subgoal[self.accepted_verbs[0]], Subgoal)\
            or isinstance(self.verb_to_subgoal[self.accepted_verbs[0]], list)
    
    def ground(self, grid_map, ids, knows_room_types=False):
        # ground to grid map
        sg = ReachRoomSubgoal(ids, self.room_type, grid_map,
                              knows_room_types=knows_room_types)
        reaches = [sg]

        # If assume knows the type of a room, then only add subgoals
        # to search in those rooms of the desired type; Otherwise,
        # add subgoal to search in all rooms. TODO: You should have
        # an update function that modifies these subgoals as observations
        # come in.
        searches = []
        for room_name in grid_map.rooms:
            if knows_room_types:
                if grid_map.rooms[room_name].room_type != self.room_type:
                    continue
            sg = SearchRoomSubgoal(ids, room_name, grid_map)
            searches.append(sg)
        self.verb_to_subgoal["Reach"] = reaches
        self.verb_to_subgoal["Search"] = searches
        self._knows_room_types = knows_room_types        
        assert self.grounded is True

    def observation_variations(self):
        """Returns a list of all possible object configurations
        for this class that are basically object observations;
        This may not be feasible in some cases."""        
        variations = []
        for room_name in self.grid_map.rooms:
            if self._knows_room_types:
                if grid_map.rooms[room_name].room_type != self.room_type:
                    continue
            objobs = RoomObservation(room_name)
            variations.append(objobs)
        return variations
    
        

class ReachRoomSubgoal(Subgoal):
    """
    Reach room type.
    Again, knows the room layout but doesn't know which
    room is which (optional)"""
    def __init__(self, ids, room_type, grid_map, knows_room_types=False):
        self.ids = ids
        self.grid_map = grid_map
        self.room_type = room_type
        self.knows_room_types = knows_room_types
        super().__init__("Reach-%s" % room_type)

    def achieve(self, state, action):
        # Achieves the goal when the robot is at the center of mass of the room
        robot_id = self.ids["Robot"]
        robot_state = state.object_states[robot_id]
        room_name = grid_map.room_of(robot_state.pose[:2])
        room = self.grid_map.rooms[room_attr.room_name]
        if self.knows_room_types:
            return room.room_type == self.room_type\
                and robot_state.pose[:2] == room.center_of_mass
        else:
            return robot_state["room_type"] == self.room_type\
                and robot_state.pose[:2] == room.center_of_mass

    def fail(self, state, action):
        return isinstance(action, Pickup)
    
    def valid(self, robot_belief):
        return True

    def effect(self, robot_belief):
        """The effect of reaching a room type is:
        Change of robot pose; Change of robot room name,
        And observing Kitchen room"""
        values = {}
        # Achieving this subgoal means the robot is at one
        # of certain room's center of mass, and it 
        for room_name in self.grid_map.rooms:
            room = self.grid_map.rooms[room_name]
            location = room.center_of_mass
            room_observation = RoomObservation(room_name, self.room_type)
            if self.knows_room_types:
                if room.room_type == self.room_type:
                    # If the robot is assumed to know the type of every room,
                    values[(self.room_type, room_observation)] = 1.0
            else:
                values[(self.room_type, room_observation)] = 1.0
        return values

    
class SearchRoomSubgoal(Subgoal):
    """Searches within a given room. Regardless of what type it is"""
    def __init__(self, ids, room_name, grid_map):
        self.ids = ids
        self.room_name = room_name
        self.grid_map = grid_map
        super().__init__("Search-%s" % room_name)

    def achieve(self, state, action):
        robot_id = self.ids["Robot"]
        robot_state = state.object_states[robot_id]
        room_name = grid_map.room_of(robot_state.pose[:2])
        if room_name != self.room_name:
            return False
        for objid in self.ids["Target"]:
            if state.object_states[objid]["is_found"]:
                return True
        return False
        
    def fail(self, state, action):
        robot_id = self.ids["Robot"]        
        robot_state = state.object_states[robot_id]
        if isinstance(action, Pickup):
            for objid in self.ids["Target"]:
                objstate = state.object_states[objid]
                if not (objstate.pose == robot_state.pose[:2]\
                        and not objstate["is_found"]):
                    return True
        return False

    def valid(self, robot_belief):
        robot_id = self.ids["Robot"]
        robot_state = robot_belief.mpe().object_states[robot_id]
        room_name = self.grid_map.room_of(robot_state.pose[:2])
        return room_name == self.room_name

    def effect(self, robot_belief):
        """The effect of finding the target is:
        Change of the target state to be found
        Observation of target; The target must be found within the room."""
        values = {}
        target_id = self.ids["Target"][0]
        target_belief = robot_belief.object_beliefs[target_id]
        for target_state in target_belief:
            if self.grid_map.room_of(target_state.pose) == self.room_name:
                observation = ItemObservation(target_state.objclass,
                                              target_state.pose)
                values[(target_state.objclass, observation)] = target_belief[target_state]
        return values    
