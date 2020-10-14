import pomdp_py
from relpomdp.home2d.tasks.common.policy_model import PolicyModel, PreferredPolicyModel
from relpomdp.home2d.domain.condition_effect import MoveEffect
from relpomdp.home2d.domain.action import *

class ExplorationActionPrior(pomdp_py.ActionPrior):
    def __init__(self, robot_id, grid_map, legal_motions,
                 num_visits_init, val_init,
                 motions={MoveE, MoveW, MoveN, MoveS}):
        self.robot_id = robot_id
        self.grid_map = grid_map
        self.legal_motions = legal_motions
        self.motions = motions
        self.num_visits_init = num_visits_init
        self.val_init = val_init

    def get_preferred_actions(self, state, history):
        """Get preferred actions. This can be used by a rollout policy as well."""
        robot_state = state.object_states[self.robot_id]
        if robot_state["pose"][:2] not in self.legal_motions:
            return set()

        cur_room = self.grid_map.room_of(robot_state["pose"][:2])
        preferences = set()

        neighbors = {MoveEffect.move_by(robot_state["pose"][:2], action.motion):action
                     for action in self.legal_motions[robot_state["pose"][:2]]}
        for next_robot_pose in neighbors:
            action = neighbors[next_robot_pose]
            if next_robot_pose[:2] not in self.grid_map.free_locations:
                preferences.add((action,
                                 self.num_visits_init, self.val_init))
        return preferences
