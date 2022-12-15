from PPOAgentsLosses import PPOAgentWithVanillaPolicyGradientLoss
from PPOAgentsLosses import PPOWithGeneralizedAdvantageLoss
from PPOAgentsLosses import PPOWithSurrogateLossWithoutClipping
from PPOAgentsLosses import PPOWithSurrogateLossWithClipping

from BasePPOAgentsEnsembleCritics import BaseEnsembleCriticsPPOAgent

class EnsembleCriticsPPOAgentWithVanillaPolicyGradientLoss(PPOAgentWithVanillaPolicyGradientLoss, BaseEnsembleCriticsPPOAgent):
    def __init__(self, actor, critic):
        super(EnsembleCriticsPPOAgentWithVanillaPolicyGradientLoss, self).__init__(actor, critic)

class EnsembleCriticsPPOWithGeneralizedAdvantageLoss(PPOWithGeneralizedAdvantageLoss, BaseEnsembleCriticsPPOAgent):
    def __init__(self, actor, critic):
        super(EnsembleCriticsPPOWithGeneralizedAdvantageLoss, self).__init__(actor, critic)

class EnsembleCriticsPPOWithSurrogateLossWithoutClipping(PPOWithSurrogateLossWithoutClipping, BaseEnsembleCriticsPPOAgent):
    def __init__(self, actor, critic):
        super(EnsembleCriticsPPOWithSurrogateLossWithoutClipping, self).__init__(actor, critic)

class EnsembleCriticsPPOWithSurrogateLossWithClipping(PPOWithSurrogateLossWithClipping, BaseEnsembleCriticsPPOAgent):
    def __init__(self, actor, critic):
        super(EnsembleCriticsPPOWithSurrogateLossWithClipping, self).__init__(actor, critic)
