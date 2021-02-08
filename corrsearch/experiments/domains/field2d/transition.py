"""Transition model"""

import pomdp_py
from corrsearch.models.robot_model import *
from corrsearch.models.state import *
from corrsearch.utils import *

class DetRobotTrans(RobotTransModel):
    """Deterministic robot transition model"""

    def __init__(self, robot_id, schema="xy"):
        self.robot_id = robot_id
        self.schema = schema

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
        else:
            next_robot_pose = robot_pose

        return RobotState(self.robot_id,
                          {"pose": next_robot_pose,
                           "loc": next_robot_pose[:2],
                           "energy": next_energy})
