from BasePPOAgentWithImages import BasePPOAgentWithImages
from PPOAgentsLosses import PPOAgentWithVanillaPolicyGradientLoss
from PPOAgentsLosses import PPOWithGeneralizedAdvantageLoss
from PPOAgentsLosses import PPOWithSurrogateLossWithoutClipping
from PPOAgentsLosses import PPOWithSurrogateLossWithClipping

class ImagesPPOAgentVanillaPolicyGradientLoss(BasePPOAgentWithImages, PPOAgentWithVanillaPolicyGradientLoss):
    def __init__(self, actor, critic):
        self.images = []
        super(ImagesPPOAgentVanillaPolicyGradientLoss, self).__init__(actor, critic)

class ImagesPPOAgentGeneralizedAdvantageLoss(BasePPOAgentWithImages, PPOWithGeneralizedAdvantageLoss):
    def __init__(self, actor, critic):
        self.images = []
        super(ImagesPPOAgentGeneralizedAdvantageLoss, self).__init__(actor, critic) 

class ImagesPPOAgentSurrogateLossWithoutClipping(BasePPOAgentWithImages, PPOWithSurrogateLossWithoutClipping):
    def __init__(self, actor, critic):
        self.images = []
        super(ImagesPPOAgentSurrogateLossWithoutClipping, self).__init__(actor, critic) 

class ImagesPPOAgentSurrogateLossWithClipping(BasePPOAgentWithImages, PPOWithSurrogateLossWithClipping):
    def __init__(self, actor, critic):
        self.images = []
        super(ImagesPPOAgentSurrogateLossWithClipping, self).__init__(actor, critic)  



    