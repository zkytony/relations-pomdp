# In this task, the robot leaves the current
# room. Assumes the robot has access to the
# room layout map.

from relpomdp.home2d.domain.action import *
from relpomdp.home2d.domain.condition_effect import CanMove, MoveEffect
from relpomdp.home2d.tasks.task import Task
from relpomdp.home2d.tasks.common.policy_model import *
import relpomdp.oopomdp.framework as oopomdp

class MoveEffectWithRoom(MoveEffect):

    def __init__(self, robot_id, grid_map):
        self.robot_id = robot_id
        self.grid_map = grid_map

    def mpe(self, state, action, byproduct=None):
        next_state = super().mpe(state, action)
        robot_state = next_state.object_states[self.robot_id]
        robot_state["room_id"] = self.grid_map.room_of(robot_state["pose"][:2]).name
        return next_state

class RewardModel(pomdp_py.RewardModel):
    def __init__(self, robot_id, room_id):
        self.robot_id = robot_id
        self.room_id = room_id

    def sample(self, state, action, next_state, **kwargs):
        return self.argmax(state, action, next_state)

    def argmax(self, state, action, next_state, **kwargs):
        """
        argmax(self, state, action, next_state, **kwargs)
        Returns the most likely reward"""
        robot_state = state.object_states[self.robot_id]
        next_robot_state = next_state.object_states[self.robot_id]
        if next_robot_state["room_id"] != self.room_id:
            if robot_state["room_id"] == self.room_id:
                return 100.0
            else:
                return -1.0
        else:
            return -1

class GreedyActionPrior(pomdp_py.ActionPrior):
    """greedy action prior for 'xy' motion scheme;
    Requires knowledge of grid map"""
    def __init__(self, robot_id, room_id, grid_map,
                 num_visits_init, val_init, motions={MoveE, MoveW, MoveN, MoveS}):
        self.robot_id = robot_id
        self.room_id = room_id
        self.grid_map = grid_map
        self.legal_motions = grid_map.compute_legal_motions(motions)
        self.motions = motions
        self.num_visits_init = num_visits_init
        self.val_init = val_init

    def get_preferred_actions(self, state, history):
        """Get preferred actions. This can be used by a rollout policy as well."""
        robot_state = state.object_states[self.robot_id]
        cur_room = robot_state["room_id"]
        preferences = set()

        if robot_state["room_id"] != self.room_id:
            neighbors = {MoveEffect.move_by(robot_state["pose"][:2], action):action
                         for action in self.legal_motions[robot_state["pose"][:2]]}
            for next_robot_pose in neighbors:
                # Prefer action to move into a different room
                action = neighbors[next_robot_pose]
                next_room = self.grid_map.room_of(next_robot_pose[:2]).name
                if next_room != cur_room:
                    preferences.add((action,
                                     self.num_visits_init, self.val_init))
        return preferences


class LeaveRoomTask(Task):
    def __init__(self,
                 robot_id,
                 room_id,
                 grid_map):
        self.robot_id = robot_id
        self.room_id = room_id
        motions = {MoveN, MoveS, MoveE, MoveW}

        cond_effects_t = []
        legal_motions = grid_map.compute_legal_motions(motions)
        cond_effects_t.append((CanMove(robot_id, legal_motions),
                               MoveEffectWithRoom(robot_id, grid_map)))
        transition_model = oopomdp.OOTransitionModel(cond_effects_t)

        observation_model = oopomdp.OOObservationModel([])
        action_prior = GreedyActionPrior(robot_id, room_id, grid_map, 10, 100)
        policy_model = PreferredPolicyModel(action_prior)
        reward_model = RewardModel(robot_id, room_id)
        super().__init__(transition_model,
                         observation_model,
                         reward_model,
                         policy_model)
