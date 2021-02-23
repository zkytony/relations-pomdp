import pomdp_py
from corrsearch.objects.object_state import JointBelief, JointState

class ThorBelief(JointBelief):
    def __init__(self,
                 robot_id, robot_belief,
                 target_id, target_belief):
        self.robot_id = robot_id
        self.target_id = target_id
        super().__init__({robot_id:robot_belief,
                          target_id:target_belief})

    def update(self, agent, observation, action):
        robot_belief = self.obj(self.robot_id)
        robot_state = robot_belief.mpe()

        robot_trans_model = agent.transition_model.robot_trans_model
        next_robot_state = robot_trans_model.sample(
            JointState({self.robot_id:robot_state}), action)
        next_robot_belief = pomdp_py.Histogram({next_robot_state:1.0})

        # compute next object belief
        target_belief = self.obj(self.target_id)
        next_target_hist = {}
        total_prob = 0.0
        for starget in target_belief:
            joint_state = JointState({self.robot_id:next_robot_state,
                                      self.target_id:starget})
            pr_obz = agent.observation_model.probability(observation,
                                                         joint_state,
                                                         action)
            next_target_hist[starget] = target_belief[starget] * pr_obz
            total_prob += next_target_hist[starget]
        # normalize
        for starget in target_belief:
            next_target_hist[starget] = next_target_hist[starget] / total_prob
        next_target_belief = pomdp_py.Histogram(next_target_hist)
        return ThorBelief(self.robot_id, next_robot_belief,
                          self.target_id, next_target_belief)
