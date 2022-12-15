from PPO import BasePPOAgent

class PPOAgentWithEnsembleCritics(BasePPOAgent):
    def __init__(self):
        super(PPOAgentWithEnsembleCritics, self).__init__()