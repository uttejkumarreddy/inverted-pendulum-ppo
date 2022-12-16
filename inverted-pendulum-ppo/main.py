import PPOAgentsLosses
import PPOAgentsEnsembleCritics
import PPOAgentsWithRecurrence
import PPOAgentsWithImages
import NeuralNets

actor = NeuralNets.ActorNN(3, 1, 64, 2)
critic = NeuralNets.CriticNN(3, 1, 64, 2)

lstm_actor = NeuralNets.ActorNN(2, 1, 64, 2)
lstm_critic = NeuralNets.CriticNN(2, 1, 64, 2)

images_actor = NeuralNets.ActorNN(3, 1, 64, 2)
images_critic = NeuralNets.CriticNN(3, 1, 64, 2)

agents = [
    PPOAgentsLosses.PPOAgentWithVanillaPolicyGradientLoss(actor, critic), # 0
    PPOAgentsLosses.PPOWithGeneralizedAdvantageLoss(actor, critic), # 1
    PPOAgentsLosses.PPOWithSurrogateLossWithoutClipping(actor, critic), # 2
    PPOAgentsLosses.PPOWithSurrogateLossWithClipping(actor, critic), # 3

    PPOAgentsEnsembleCritics.EnsembleCriticsPPOAgentWithVanillaPolicyGradientLoss(actor, critic), # 4
    PPOAgentsEnsembleCritics.EnsembleCriticsPPOWithGeneralizedAdvantageLoss(actor, critic), # 5
    PPOAgentsEnsembleCritics.EnsembleCriticsPPOWithSurrogateLossWithoutClipping(actor, critic), # 6
    PPOAgentsEnsembleCritics.EnsembleCriticsPPOWithSurrogateLossWithClipping(actor, critic), # 7

    PPOAgentsWithRecurrence.RecurrencePPOAgentWithVanillaPolicyGradientLoss(lstm_actor, lstm_critic), # 8
    PPOAgentsWithRecurrence.RecurrencePPOWithGeneralizedAdvantageLoss(lstm_actor, lstm_critic), # 9
    PPOAgentsWithRecurrence.RecurrencePPOWithSurrogateLossWithoutClipping(lstm_actor, lstm_critic), # 10
    PPOAgentsWithRecurrence.RecurrencePPOWithSurrogateLossWithClipping(lstm_actor, lstm_critic), # 11

    PPOAgentsWithImages.ImagesPPOAgentVanillaPolicyGradientLoss(images_actor, images_critic), # 12
    PPOAgentsWithImages.ImagesPPOAgentGeneralizedAdvantageLoss(images_actor, images_critic), # 13
    PPOAgentsWithImages.ImagesPPOAgentSurrogateLossWithoutClipping(images_actor, images_critic), # 14
    PPOAgentsWithImages.ImagesPPOAgentSurrogateLossWithClipping(images_actor, images_critic), # 15
]

agent = agents[12]
agent.train()
agent.plot_episodic_rewards()
agent.plot_episodic_losses()