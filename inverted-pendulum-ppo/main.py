import PPOAgentsLosses
import PPOAgentsEnsembleCritics

agents = [
    PPOAgentsLosses.PPOAgentWithVanillaPolicyGradientLoss(), # 0
    PPOAgentsLosses.PPOWithGeneralizedAdvantageLoss(), # 1
    PPOAgentsLosses.PPOWithSurrogateLossWithoutClipping(), # 2
    PPOAgentsLosses.PPOWithSurrogateLossWithClipping(), # 3

    PPOAgentsEnsembleCritics.EnsembleCriticsPPOAgentWithVanillaPolicyGradientLoss(), # 4
    PPOAgentsEnsembleCritics.EnsembleCriticsPPOWithGeneralizedAdvantageLoss(), # 5
    PPOAgentsEnsembleCritics.EnsembleCriticsPPOWithSurrogateLossWithoutClipping(), # 6
    PPOAgentsEnsembleCritics.EnsembleCriticsPPOWithSurrogateLossWithClipping(), # 7
]

agent = agents[7]
agent.train()
agent.plot_episodic_rewards()
agent.plot_episodic_losses()