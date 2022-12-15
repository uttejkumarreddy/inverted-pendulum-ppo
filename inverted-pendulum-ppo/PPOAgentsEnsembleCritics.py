import PPOAgentsLosses

class EnsembleCriticsPPOAgentWithVanillaPolicyGradientLoss(PPOAgentsLosses.PPOAgentWithVanillaPolicyGradientLoss):
    def __init__(self):
        super(EnsembleCriticsPPOAgentWithVanillaPolicyGradientLoss, self).__init__()

class EnsembleCriticsPPOWithGeneralizedAdvantageLoss(PPOAgentsLosses.PPOWithGeneralizedAdvantageLoss):
    def __init__(self):
        super(EnsembleCriticsPPOWithGeneralizedAdvantageLoss, self).__init__()

class EnsembleCriticsPPOWithSurrogateLossWithoutClipping(PPOAgentsLosses.PPOWithSurrogateLossWithoutClipping):
    def __init__(self):
        super(EnsembleCriticsPPOWithSurrogateLossWithoutClipping, self).__init__()

class EnsembleCriticsPPOWithSurrogateLossWithClipping(PPOAgentsLosses.PPOWithSurrogateLossWithClipping):
    def __init__(self):
        super(EnsembleCriticsPPOWithSurrogateLossWithClipping, self).__init__()