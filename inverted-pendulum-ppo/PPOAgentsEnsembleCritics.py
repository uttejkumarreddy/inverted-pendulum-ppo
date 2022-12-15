import PPOAgentsLosses

class EnsembleCriticsPPOAgentWithVanillaPolicyGradientLoss(PPOAgentsLosses.PPOAgentWithVanillaPolicyGradientLoss):
    def __init__(self, actor, critic):
        super(EnsembleCriticsPPOAgentWithVanillaPolicyGradientLoss, self).__init__(actor, critic)

class EnsembleCriticsPPOWithGeneralizedAdvantageLoss(PPOAgentsLosses.PPOWithGeneralizedAdvantageLoss):
    def __init__(self, actor, critic):
        super(EnsembleCriticsPPOWithGeneralizedAdvantageLoss, self).__init__(actor, critic)

class EnsembleCriticsPPOWithSurrogateLossWithoutClipping(PPOAgentsLosses.PPOWithSurrogateLossWithoutClipping):
    def __init__(self, actor, critic):
        super(EnsembleCriticsPPOWithSurrogateLossWithoutClipping, self).__init__(actor, critic)

class EnsembleCriticsPPOWithSurrogateLossWithClipping(PPOAgentsLosses.PPOWithSurrogateLossWithClipping):
    def __init__(self, actor, critic):
        super(EnsembleCriticsPPOWithSurrogateLossWithClipping, self).__init__(actor, critic)