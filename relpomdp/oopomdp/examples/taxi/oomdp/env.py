# The Taxi domain
#
# The coordinate system of Taxi is: +x is to the right, +y is up, origin is bottom-left.
# the wall is on the right side of the grid cell.

import pomdp_py
from relpomdp.oopomdp.framework import *
from relpomdp.oopomdp.examples.taxi.actions import *
from relpomdp.oopomdp.examples.taxi.visual import *

# Object classes and attributes
class PoseState(ObjectState):
    @property
    def pose(self):
        return self["pose"]

class TaxiState(PoseState):
    CLASS = "Taxi"
    def __init__(self, pose):
        super().__init__(TaxiState.CLASS,
                         {"pose": pose})
    def copy(self):
        return TaxiState(self["pose"])

class PassengerState(PoseState):
    CLASS = "Passenger"
    def __init__(self, pose, in_taxi):
        super().__init__(PassengerState.CLASS,
                         {"pose": pose,
                          "in_taxi": in_taxi})
    def copy(self):
        return PassengerState(self["pose"], self["in_taxi"])

class WallState(PoseState):
    CLASS = "Wall"
    def __init__(self, pose, direction):
        """direction can be 'H' or 'V'"""
        super().__init__(WallState.CLASS,
                         {"pose": pose,
                          "direction": direction})
    def copy(self):
        return WallState(self["pose"], self.direction)

    @property
    def direction(self):
        return self["direction"]

class DestinationState(PoseState):
    CLASS = "Destination"
    def __init__(self, pose):
        super().__init__(DestinationState.CLASS,
                         {"pose": pose})
    def copy(self):
        return DestinationState(self["pose"])                

class DomainState(OOState):
    def __init__(self, taxi_id, psgr_id, dst_id, object_states):
        self.taxi_id = taxi_id
        self.psgr_id = psgr_id
        self.dst_id = dst_id        
        super().__init__(object_states)

    @property
    def taxi_state(self):
        return self.object_states[self.taxi_id]

    @property    
    def passenger_state(self):
        return self.object_states[self.psgr_id]

    @property    
    def destination_state(self):
        return self.object_states[self.dst_id]    

    def copy(self):
        object_states = {objid : self.object_states[objid].copy()
                         for objid in self.object_states}
        return DomainState(self.taxi_id, self.psgr_id, self.dst_id,
                           object_states)
    

# Relations
class Touch(Relation):
    def __init__(self, direction, class1, class2):
        if direction not in {"N", "E", "S", "W"}:
            raise ValueError("Invalid direction %s" % direction)
        self.direction = direction
        super().__init__("touch-%s" % direction,
                         class1, class2)
        
    def eval(self, object_state1, object_state2):
        """Returns True if the Relation holds. False otherwise.
        According to the paper, touch(o1,o2) holds if
        o2 is exactly one cell North, South, East, or West of o1.

        Note that we assume a vertical wall is on the right edge
        of a grid cell and a horizontal wall is on the top edge.
        Touching a wall means having a wall on any of the edges
        that the object_state1 is in."""
        if not isinstance(object_state2, WallState):
            raise ValueError("Taxi domain (at least in OO-MDP) requires"
                             "the Touch relation to involve Wall as the second object.")
        x1, y1 = object_state1.pose
        x2, y2 = object_state2.pose
        if self.direction == "N":
            if object_state2.direction == "H":
                return x1 == x2 and y1 == y2
            else:
                return False  # vertical wall cannot touch at North
        elif self.direction == "S":
            if object_state2.direction == "H":
                return x1 == x2 and y1 == y2 + 1
            else:
                return False  # vertical wall cannot touch at North
        elif self.direction == "E":
            if object_state2.direction == "V":
                return x1 == x2 and y1 == y2
            else:
                return False  # vertical wall cannot touch at East
        else:
            assert self.direction == "W"
            if object_state2.direction == "V":
                return x1 == x2 + 1 and y1 == y2
            else:
                return False  # vertical wall cannot touch at East
        
class On(Relation):
    def __init__(self, class1, class2):
        super().__init__("on",
                         class1, class2)
        
    def eval(self, object_state1, object_state2):
        """Returns True if the Relation holds. False otherwise.
        According to the paper, on(o1,o2) holds if o1 and o2 are
        overlapping"""
        return object_state1.pose == object_state2.pose

touch_N_taxi_wall = Touch("N", TaxiState.CLASS, WallState.CLASS)
touch_S_taxi_wall = Touch("S", TaxiState.CLASS, WallState.CLASS)
touch_E_taxi_wall = Touch("E", TaxiState.CLASS, WallState.CLASS)
touch_W_taxi_wall = Touch("W", TaxiState.CLASS, WallState.CLASS)
on_taxi_passenger = On(TaxiState.CLASS, PassengerState.CLASS)
on_taxi_destination = On(TaxiState.CLASS, DestinationState.CLASS)


# Conditions and effects
class CanMove(Condition):
    # Strictly speaking OO-MDP expects conditions like CanMoveE, CanMoveN, etc.
    # but that is too verbose
    def satisfy(self, state, action):
        if not isinstance(action, Move):
            return False
        taxi = state.taxi_state
        for objid in state.object_states:
            if isinstance(state.object_states[objid], WallState):
                wall = state.object_states[objid]
                if touch_N_taxi_wall(taxi, wall) and action == MoveN:
                    return False
                elif touch_S_taxi_wall(taxi, wall) and action == MoveS:
                    return False
                elif touch_E_taxi_wall(taxi, wall) and action == MoveE:
                    return False
                elif touch_W_taxi_wall(taxi, wall) and action == MoveW:
                    return False
        return True

class MoveEffect(DeterministicEffect):
    """Deterministically move"""
    def __init__(self):
        super().__init__("arithmetic")  # ? not really a reason to name the type this way
        
    def mpe(self, state, action):
        """Returns an OOState after applying this effect on `state`"""
        next_state = state.copy()
        next_state.taxi_state["pose"] = (state.taxi_state["pose"][0] + action.motion[0],
                                         state.taxi_state["pose"][1] + action.motion[1])
        if next_state.passenger_state["in_taxi"]:
            next_state.passenger_state["pose"] = next_state.taxi_state["pose"]
        return next_state

class CanPickup(Condition):
    def satisfy(self, state, action):
        if not isinstance(action, Pickup):
            return False
        taxi = state.taxi_state
        passenger = state.passenger_state
        return on_taxi_passenger(taxi, passenger) and passenger["in_taxi"] == False
    
class PickupEffect(DeterministicEffect):
    """Deterministically move"""
    def __init__(self):
        super().__init__("constant")  # ? not really a reason to name the type this way
        
    def mpe(self, state, action):
        """Returns an OOState after applying this effect on `state`"""
        next_state = state.copy()
        next_state.passenger_state["in_taxi"] = True
        return next_state

class CanDropoff(Condition):
    def satisfy(self, state, action):
        if not isinstance(action, Dropoff):
            return False        
        taxi = state.taxi_state
        passenger = state.passenger_state
        return on_taxi_passenger(taxi, passenger) and passenger["in_taxi"] == True

class DropoffEffect(DeterministicEffect):
    """Deterministically move"""
    def __init__(self):
        super().__init__("constant")  # ? not really a reason to name the type this way
        
    def mpe(self, state, action):
        """Returns an OOState after applying this effect on `state`"""
        next_state = state.copy()
        next_state.passenger_state["in_taxi"] = False
        return next_state

class RewardModel(pomdp_py.RewardModel):
    def sample(self, state, action, next_state, **kwargs):
        return self.argmax(state, action, next_state)
    
    def argmax(self, state, action, next_state, **kwargs):
        """
        argmax(self, state, action, next_state, **kwargs)
        Returns the most likely reward"""
        # Reward is 1 if task finished, -0.1 otherwise. -1 if wrong.
        taxi = state.taxi_state
        destination = state.destination_state
        passenger = state.passenger_state
        if action == Dropoff():
            if on_taxi_destination(state.taxi_state,
                                   state.destination_state):
                if state.passenger_state["in_taxi"]:
                    # passenger was in taxi, taxi is on destination, and dropoff,
                    return 1.0
                
            return -1.0  # bad drop off
        
        else:
            return -0.1

class TaxiEnvironment(OOEnvironment):
    def __init__(self,
                 width, length,
                 init_taxi_pose,
                 init_passenger_pose,
                 init_destination_pose,
                 walls):

        self.width = width
        self.length = length
        
        taxi_state = TaxiState(init_taxi_pose)
        passenger_state = PassengerState(init_passenger_pose, False)
        destination_state = DestinationState(init_destination_pose)

        taxi_id = 10
        passenger_id = 1
        destination_id = 100
        wall_states = {}
        for i in range(len(walls)):
            x, y, direction = walls[i]
            wall_states[i + 200] = WallState((x,y), direction=direction)

        init_state = DomainState(taxi_id, passenger_id, destination_id,
                                 {taxi_id: taxi_state,
                                  passenger_id: passenger_state,
                                  destination_id: destination_state,
                                  **wall_states})
        relations = {touch_N_taxi_wall,
                     touch_S_taxi_wall,
                     touch_E_taxi_wall,
                     touch_W_taxi_wall,
                     on_taxi_passenger,
                     on_taxi_destination}
        cond_effects = {(CanMove(), MoveEffect()),
                        (CanPickup(), PickupEffect()),
                        (CanDropoff(), DropoffEffect())}
        reward_model = RewardModel()
        super().__init__(init_state, relations, cond_effects, reward_model)


if __name__ == "__main__":
    top_walls = [(x,4,"H") for x in range(5)]
    bottom_walls = [(x,-1,"H") for x in range(5)]
    left_walls = [(-1,y,"V") for y in range(5)]
    right_walls = [(4,y,"V") for y in range(5)]
    inner_walls = [(0,0,"V"), (0,1,"V"), (1,3,"V"), (1,4,"V"), (2,0,"V"), (2,1,"V")]
    
    env = TaxiEnvironment(5, 5,
                          (2,4),
                          (0,0),
                          (4,4),
                          top_walls + bottom_walls + left_walls + right_walls\
                          + inner_walls)
    viz = TaxiViz(env, controllable=True)
    viz.on_init()
    viz.on_execute()
