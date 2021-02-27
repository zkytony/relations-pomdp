
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


class TopoMove(Move):
    def __init__(self, src, dst, src_nid=None, dst_nid=None, energy_cost=0.0):
        """
        src, dst are locations in POMDP grid
        """
        self.src = src
        self.dst = dst
        self.src_nid = src_nid
        self.dst_nid = dst_nid
        delta = (dst[0] - src[0],
                 dst[1] - src[1],
                 0)
        super().__init__(delta,
                         "move(%d->%d)" % (src_nid, dst_nid),
                         energy_cost=energy_cost)

class TopoRobotTrans(RobotTransModel):
    def __init__(self, robot_id, topo_map, grid_map, grid_size=0.25):
        self.robot_id = robot_id
        self.topo_map = topo_map
        self.grid_map = grid_map
        self.grid_size = grid_size
        self.schema = None

        # Create a mapping from grid pos to topo map node id
        self._grid_nid_map = {}
        for nid in self.topo_map.nodes:
            thor_x, thor_z = self.topo_map.nodes[nid].pose
            x, y = self.grid_map.to_grid_pos(thor_x, thor_z,
                                             grid_size=self.grid_size, avoid_obstacle=True)
            self._grid_nid_map[(x,y)] = nid

    def move_by(self, robot_pose, action):
        """action can be TopoMove, or a Move action with rotation.
        If former, assume that robot_pose is at the source of the action;
        if not, will raise exception"""
        assert self.schema == "topo"
        rx, ry, rth = robot_pose
        if isinstance(action, TopoMove):
            if (rx, ry) != action.src:
                raise ValueError("Robot (at {}) not at the source of TopoMove (at {})\nInvalid move"\
                                 .format((rx, ry), action.src))
            dx, dy, _ = action.delta
            return rx + dx, ry + dy, rth
        else:
            # This must be a Move action for schema vw.
            forward, angle = action.delta
            assert forward == 0.0, "Not supporting forward in TopoRobotTrans"
            new_rth = rth + angle  # angle (radian)
            new_rth = new_rth % (2*math.pi)
            return rx, ry, new_rth

    def probability(self, next_robot_state, state, action, **kwargs):
        """
        Pr(s_r' | s, a)
        """
        # if isinstance(action, TopoMove) and action.src_nid == 12 and action.dst_nid == 11:
        #     import pdb; pdb.set_trace()
        return indicator(next_robot_state == self.sample(state, action))

    def sample(self, state, action, **kwargs):
        """
        s_r' ~ T(s,a)
        """
        robot_state = state[self.robot_id]
        robot_pose = state[self.robot_id]["pose"]

        # Enforce snapping onto topo node
        if robot_pose[:2] not in self._grid_nid_map:
            robot_loc = min(self._grid_nid_map.keys(), key=lambda p: euclidean_dist(p, robot_pose[:2]))
            robot_pose = (*robot_loc, robot_pose[2])

        next_energy = robot_state["energy"] - action.energy_cost
        if isinstance(action, Move):
            next_robot_pose = self.move_by(robot_pose, action)
            if next_robot_pose[:2] not in self.grid_map.free_locations:
                import pdb; pdb.set_trace()
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

class TopoPolicyModel(pomdp_py.RolloutPolicy):
    """This should incorporate BasicPolicyModel with topological moves"""
    def __init__(self, robot_id, topo_map, grid_map,
                 rotate_actions, detect_actions, declare_actions, grid_size=0.25):
        self.robot_id = robot_id
        self.grid_size = grid_size
        self._motion_map = {}
        for nid in topo_map.nodes:
            move_actions = set()
            thor_x, thor_z = topo_map.nodes[nid].pose
            src_pos = grid_map.to_grid_pos(thor_x, thor_z, grid_size=self.grid_size, avoid_obstacle=True)
            for neighbor_nid in topo_map.neighbors(nid):
                dst_thor_x, dst_thor_z = topo_map.nodes[neighbor_nid].pose
                dst_pos = grid_map.to_grid_pos(dst_thor_x, dst_thor_z, grid_size=self.grid_size, avoid_obstacle=True)
                move_actions.add(TopoMove(src_pos, dst_pos, nid, neighbor_nid,
                                          energy_cost=0.8*euclidean_dist(src_pos, dst_pos)))
            self._motion_map[src_pos] = move_actions
        self.detect_actions = detect_actions
        self.declare_actions = declare_actions
        self.move_actions = set().union(*[self._motion_map[src_pos]
                                          for src_pos in self._motion_map]) | rotate_actions
        self.rotate_actions = rotate_actions
        self.actions = move_actions | declare_actions | detect_actions

    def get_all_actions(self, state, history=None):
        """If the last action is a move, then this action will not be a move.
        (Domain-specific setting)"""
        if state is None:
            return self.actions
        else:
            moves = self.valid_moves(state)
            if history is None or len(history) == 0:
                return self.detect_actions
            else:
                last_action = history[-1][0]
                if isinstance(last_action, UseDetector):
                    return moves | self.detect_actions | self.declare_actions
                else:
                    return moves | self.detect_actions

    def rollout(self, state, history=None):
        return random.sample(self.get_all_actions(state=state, history=history), 1)[0]

    def sample(self, state, history=None):
        return random.sample(self._get_all_actions(state=state, history=history), 1)[0]

    def valid_moves(self, state):
        # Only declare if the last one was a UseDetector
        robot_loc = state[self.robot_id].loc
        if robot_loc not in self.move_actions:
            # Snap robot loc to closest node
            robot_loc = min(self._motion_map.keys(),
                            key=lambda p: euclidean_dist(p, robot_loc))
        moves = self._motion_map[robot_loc] | self.rotate_actions
        return moves
