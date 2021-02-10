"""Transition model"""

import pomdp_py
from corrsearch.models.robot_model import *
from corrsearch.models.state import *
from corrsearch.utils import *

class DetRobotTrans(RobotTransModel):
    """Deterministic robot transition model"""

    def __init__(self, robot_id, locations, schema="xy", actions=None):
        """
        If `actions` is supplied, then robot state can be explicitly enumerated
        """
        self.robot_id = robot_id
        self.schema = schema
        self.locations = set(locations)
        self.actions = actions

    def move_by(self, robot_pose, action):
        if self.schema == "xy":
            dx, dy, th = action.delta
            rx, ry = robot_pose[:2]
            return (rx + dx, ry + dy, th)

    def probability(self, next_robot_state, state, action, **kwargs):
        """
        Pr(s_r' | s, a)
        """
        return indicator(next_robot_state == self.sample(state, action))

    def sample(self, state, action, **kwargs):
        """
        s_r' ~ T(s,a)
        """
        robot_state = state[self.robot_id]
        robot_pose = state[self.robot_id]["pose"]
        next_energy = robot_state["energy"] - action.energy_cost
        if isinstance(action, Move):
            next_robot_pose = self.move_by(robot_pose, action)
            if next_robot_pose[:2] not in self.locations:
                next_robot_pose = robot_pose
        else:
            next_robot_pose = robot_pose

        return RobotState(self.robot_id,
                          {"pose": next_robot_pose,
                           "loc": next_robot_pose[:2],
                           "energy": next_energy})

    def get_all_states(self):
        """The set of robot states is the set of
        all possible robot poses. We will ignore the
        `energy` attribute."""
        print("Warning: energy attribute ignored when enumerating robot states")
        if self.schema != "xy":
            raise ValueError("Currently only support explicit enumeration for 'xy' schema")
        angles = set()
        for a in self.actions:
            assert a.energy_cost == 0.0,\
                "Unsupported action energy cost for explicit enumeration of state space."
            if isinstance(a, Move):
                # by definition of 'xy' schema, th is an absolute angle
                # the the robot will take after applying this action
                dx, dy, th = a.delta
                angles.add(th)

        # enumerate over all poses
        robot_states = []
        for loc in self.locations:
            for th in angles:
                pose = (loc[0], loc[1], th)
                robot_state = RobotState(self.robot_id,
                                         {"loc": loc,
                                          "pose": pose,
                                          "energy": 0.0})
                robot_states.append(robot_state)
        return robot_states
