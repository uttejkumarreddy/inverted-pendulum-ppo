from PPOWithGeneralizedAdvantageLoss import PPOWithGeneralizedAdvantageLoss
from PPOWithSurrogateLossWithClipping import PPOWithSurrogateLossWithClipping
from PPOWithSurrogateLossWithoutClipping import PPOWithSurrogateLossWithoutClipping
from PPOWithVanillaPolicyGradientLoss import PPOAgentWithVanillaPolicyGradientLoss

agents = [
    PPOWithGeneralizedAdvantageLoss(),
    PPOWithSurrogateLossWithClipping(),
    PPOWithSurrogateLossWithoutClipping(),
    PPOAgentWithVanillaPolicyGradientLoss()
]

agent = agents[0]
agent.train()
agent.plot_episodic_losses()
agent.plot_episodic_rewards()