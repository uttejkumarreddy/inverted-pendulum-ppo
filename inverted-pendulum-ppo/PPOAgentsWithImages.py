from BasePPOAgentWithImages import BasePPOAgentWithImages
from PPOAgentsLosses import PPOAgentWithVanillaPolicyGradientLoss
from PPOAgentsLosses import PPOWithGeneralizedAdvantageLoss
from PPOAgentsLosses import PPOWithSurrogateLossWithoutClipping
from PPOAgentsLosses import PPOWithSurrogateLossWithClipping

class ImagesPPOAgentVanillaPolicyGradientLoss(BasePPOAgentWithImages, PPOAgentWithVanillaPolicyGradientLoss):
    def __init__(self, actor, critic):
        super(ImagesPPOAgentVanillaPolicyGradientLoss, self).__init__(actor, critic)

class ImagesPPOAgentGeneralizedAdvantageLoss(BasePPOAgentWithImages, PPOWithGeneralizedAdvantageLoss):
    def __init__(self, actor, critic):
        super(ImagesPPOAgentGeneralizedAdvantageLoss, self).__init__(actor, critic) 

class ImagesPPOAgentSurrogateLossWithoutClipping(BasePPOAgentWithImages, PPOWithSurrogateLossWithoutClipping):
    def __init__(self, actor, critic):
        super(ImagesPPOAgentSurrogateLossWithoutClipping, self).__init__(actor, critic) 

class ImagesPPOAgentSurrogateLossWithClipping(BasePPOAgentWithImages, PPOWithSurrogateLossWithClipping):
    def __init__(self, actor, critic):
        super(ImagesPPOAgentSurrogateLossWithClipping, self).__init__(actor, critic)  



    