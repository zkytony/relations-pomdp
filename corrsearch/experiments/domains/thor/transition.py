
import pomdp_py
from corrsearch.models.robot_model import *
from corrsearch.models.state import *
from corrsearch.models.transition import *
from corrsearch.utils import *

class DetRobotTrans(RobotTransModel):
    """Deterministic robot transition model
    Don't confuse this with RobotModel."""

    def __init__(self, robot_id, grid_map, schema="vw"):
        self.robot_id = robot_id
        self.grid_map = grid_map
        self.schema = schema

    def move_by(self, robot_pose, action):
        rx, ry, rth = robot_pose
        if self.schema == "vw":
            forward, angle = action.delta
            new_rth = rth + angle  # angle (radian)
            new_rx = int(round(rx + forward*math.cos(new_rth)))
            new_ry = int(round(ry + forward*math.sin(new_rth)))
            new_rth = new_rth % (2*math.pi)
        return (new_rx, new_ry, new_rth)

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
            if next_robot_pose[:2] not in self.grid_map.free_locations:
                next_robot_pose = robot_pose
        else:
            next_robot_pose = robot_pose

        terminal = False
        if isinstance(action, Declare):
            terminal = True

        return RobotState(self.robot_id,
                          {"pose": next_robot_pose,
                           "loc": next_robot_pose[:2],
                           "energy": next_energy,
                           "terminal": terminal})
