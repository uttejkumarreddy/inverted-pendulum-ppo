from BasePPOAgentRecurrence import BasePPOAgentRecurrence
from PPOAgentsLosses import PPOAgentWithVanillaPolicyGradientLoss
from PPOAgentsLosses import PPOWithGeneralizedAdvantageLoss
from PPOAgentsLosses import PPOWithSurrogateLossWithoutClipping
from PPOAgentsLosses import PPOWithSurrogateLossWithClipping

class RecurrencePPOAgentWithVanillaPolicyGradientLoss(BasePPOAgentRecurrence, PPOAgentWithVanillaPolicyGradientLoss):
    def __init__(self, actor, critic):
        super(RecurrencePPOAgentWithVanillaPolicyGradientLoss, self).__init__(actor, critic)

class RecurrencePPOWithGeneralizedAdvantageLoss(BasePPOAgentRecurrence, PPOWithGeneralizedAdvantageLoss):
    def __init__(self, actor, critic):
        super(RecurrencePPOWithGeneralizedAdvantageLoss, self).__init__(actor, critic)

class RecurrencePPOWithSurrogateLossWithoutClipping(BasePPOAgentRecurrence, PPOWithSurrogateLossWithoutClipping):
    def __init__(self, actor, critic):
        super(RecurrencePPOWithSurrogateLossWithoutClipping, self).__init__(actor, critic)

class RecurrencePPOWithSurrogateLossWithClipping(BasePPOAgentRecurrence, PPOWithSurrogateLossWithClipping):
    def __init__(self, actor, critic):
        super(RecurrencePPOWithSurrogateLossWithClipping, self).__init__(actor, critic)