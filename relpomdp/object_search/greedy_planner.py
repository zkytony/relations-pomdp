import pomdp_py
import random
from collections import deque
from relpomdp.object_search.state import *
from relpomdp.object_search.action import *
from relpomdp.object_search.observation import *
from relpomdp.object_search.condition_effect import *

class ManualPlanner(pomdp_py.Planner):
    """ONLY FOR SEARCHER"""
    def __init__(self, ids):
        self.ids = ids
        self._pickup_next = False
        self.pickup_cond = CanPickup(ids)
        
    def update(self, agent, *args, **kwargs):
        mpe_state = agent.belief.mpe()
        if self.pickup_cond.satisfy(mpe_state, Pickup()):
            self._pickup_next = True
        else:
            self._pickup_next = False

class GreedyPlanner(ManualPlanner):
    # Greedily moves to the location of highest belief,
    # and look around. Take "Find" after seeing an object.
    def __init__(self, ids):
        super().__init__(ids)
        self._actions = deque([])

    def plan(self, agent):
        if self._pickup_next:
            self._pickup_next = False
            return Pickup()
        else:
            motion_policy = agent.policy_model.motion_policy
            if len(self._actions) == 0:
                # make new path
                mpe_state = agent.belief.mpe()
                agent_state = mpe_state.object_states[self.ids["Robot"]]
                agent_pose = agent_state.pose
                ## obtain the pose of an undetected object
                for objid in mpe_state.object_states:
                    if objid != self.ids["Robot"]\
                       and not mpe_state.object_states[objid].is_found:
                        object_pose = mpe_state.object_states[objid].pose
                        motion_actions =\
                            motion_policy.path_between(
                                agent_pose, object_pose,
                                return_actions=True)
                        if len(motion_actions[:-1]) > 0:
                            self._actions.extend(motion_actions)
                            break                        
                if len(self._actions) == 0:
                    print("Warning: GreedyPlanner plan does not move the robot."\
                          "so the robot will move randomly")
                    valid_motions =\
                        motion_policy.valid_motions(agent_pose)
                    action = random.sample(valid_motions, 1)[0]
                    self._actions.append(action)
            # Return action
            return self._actions.popleft()
