import pomdp_py

class Task:

    """
    A Task is essentially an agent but without belief;
    that is, it has all the other components: T, R, O. pi.

    Additionally, a Task defines a function: s -> Dist(s')
    where s is a given state, Dist(s') is a distribution over
    the states assuming the task is completed.
    """

    def __init__(self,
                 transition_model,
                 observation_model,
                 reward_model,
                 policy_model):
        self.transition_model = transition_model
        self.observation_model = observation_model
        self.reward_model = reward_model
        self.policy_model = policy_model

    def effect(self, state, *args, **kwargs):
        """
        Given a state, returns a distribution over
        the resulting state once this task is complete.
        """
        raise NotImplementedError

    def to_agent(self, belief):
        return pomdp_py.Agent(belief,
                              self.policy_model,
                              transition_model=self.transition_model,
                              observation_model=self.observation_model,
                              reward_model=self.reward_model)
