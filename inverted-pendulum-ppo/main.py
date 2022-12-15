import PPOAgentsLosses
import PPOAgentsEnsembleCritics
import NeuralNets

actor = NeuralNets.ActorNN(3, 1, 64, 2)
critic = NeuralNets.CriticNN(3, 1, 64, 2)

lstm_actor = NeuralNets.ActorNN(3, 1, 64, 2)
lstm_critic = NeuralNets.CriticNN(3, 1, 64, 2)

agents = [
    PPOAgentsLosses.PPOAgentWithVanillaPolicyGradientLoss(actor, critic), # 0
    PPOAgentsLosses.PPOWithGeneralizedAdvantageLoss(actor, critic), # 1
    PPOAgentsLosses.PPOWithSurrogateLossWithoutClipping(actor, critic), # 2
    PPOAgentsLosses.PPOWithSurrogateLossWithClipping(actor, critic), # 3

    PPOAgentsEnsembleCritics.EnsembleCriticsPPOAgentWithVanillaPolicyGradientLoss(lstm_actor, lstm_critic), # 4
    PPOAgentsEnsembleCritics.EnsembleCriticsPPOWithGeneralizedAdvantageLoss(lstm_actor, lstm_critic), # 5
    PPOAgentsEnsembleCritics.EnsembleCriticsPPOWithSurrogateLossWithoutClipping(lstm_actor, lstm_critic), # 6
    PPOAgentsEnsembleCritics.EnsembleCriticsPPOWithSurrogateLossWithClipping(lstm_actor, lstm_critic), # 7
]

agent = agents[7]
agent.train()
agent.plot_episodic_rewards()
agent.plot_episodic_losses()