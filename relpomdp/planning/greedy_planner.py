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


class RandomPlanner(ManualPlanner):
    # Greedily moves to the location of highest belief,
    # and look around. Take "Find" after seeing an object.
    def __init__(self, ids):
        super().__init__(ids)

    def plan(self, agent):
        if self._pickup_next:
            self._pickup_next = False
            return Pickup()
        else:
            motion_policy = agent.policy_model.motion_policy
            agent_state = agent.belief.mpe().object_states[self.ids["Robot"]]
            valid_motions = motion_policy.valid_motions(agent_state.pose)
            return random.sample(valid_motions, 1)[0]
        


    def path_between(self, position1, position2, return_actions=True):
        """Note that for the return_actions=True case to return reasonable
        actions, the motion actions scheme needs to be `xy`, i.e. along the axes"""
        # Finds a path between position1 and position2.
        # Uses the Dijkstra's algorithm.
        V = set({(x,y)    # all valid positions
                 for x in range(self._grid_map.width) 
                 for y in range(self._grid_map.length)
                 if self._grid_map.within_bounds((x,y))})
        position1 = position1[:2]  # If it is robot pose then it has length 3.
        S = set({})
        d = {v:float("inf")
             for v in V
             if v != position1}
        d[position1] = 0
        prev = {position1: None}
        while len(S) < len(V):
            diff_set = V - S
            v = min(diff_set, key=lambda v: d[v])
            S.add(v)
            neighbors = self.get_neighbors(v)
            for w in neighbors:
                motion_action = neighbors[w]
                cost_vw = motion_action.distance_cost
                if d[v] + cost_vw < d[w[:2]]:
                    d[w[:2]] = d[v] + cost_vw
                    prev[w[:2]] = (v, motion_action)

        # Return a path
        path = []
        pair = prev[position2[:2]]
        if pair is None:
            if not return_actions:
                path.append(position2)
        else:
            while pair is not None:
                position, action = pair
                if return_actions:
                    # Return a sequence of actions that moves the robot from position1 to position2.
                    path.append(action)
                else:
                    # Returns the grid cells along the path
                    path.append(position)
                pair = prev[position]
        return list(reversed(path))

    def get_neighbors(self, robot_pose):
        neighbors = {MoveEffect.move_by(robot_pose, action.motion):action
                     for action in self.valid_motions(robot_pose)}
        return neighbors
        
        
        
