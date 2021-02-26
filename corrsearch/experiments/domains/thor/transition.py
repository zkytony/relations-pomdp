
import pomdp_py
from corrsearch.models.robot_model import *
from corrsearch.models.state import *
from corrsearch.models.transition import *
from corrsearch.utils import *

class DetRobotTrans(RobotTransModel):
    """Deterministic robot transition model
    Don't confuse this with RobotModel."""

    def __init__(self, robot_id, grid_map):
        self.robot_id = robot_id
        self.grid_map = grid_map
        self.schema = None

    def move_by(self, robot_pose, action):
        """Note: agent by default (0 angle) looks in the +z direction in Unity,
        which corresponds to +y here.That's why I'm multiplying y with cos."""
        rx, ry, rth = robot_pose
        if self.schema == "vw":
            forward, angle = action.delta
            new_rth = rth + angle  # angle (radian)
            new_rx = int(round(rx + forward*math.sin(new_rth)))
            new_ry = int(round(ry + forward*math.cos(new_rth)))
            new_rth = new_rth % (2*math.pi)
        elif self.schema == "xy":
            dx, dy, th = action.delta
            rx, ry = robot_pose[:2]
            new_rx = rx + dx
            new_ry = ry + dy
            new_rth = th
        else:
            raise ValueError("Invalid schema: ", self.schema)
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


class TopoRobotTrans(RobotTransModel):
    pass
