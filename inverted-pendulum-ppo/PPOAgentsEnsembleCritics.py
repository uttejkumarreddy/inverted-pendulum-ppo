import torch

from BasePPOAgent import BasePPOAgent
from NeuralNets import CriticNN
from torch.optim import Adam

class EnsembleCriticsPPOAgent(BasePPOAgent):
    def __init__(self):
        super(EnsembleCriticsPPOAgent, self).__init__()

        self.critic2 = CriticNN(self.input_size, self.output_size, 8, self.n_layers)
        self.critic3 = CriticNN(self.input_size, self.output_size, 32, self.n_layers)

        self.critic2_optimizer = Adam(self.critic2.parameters(), lr = self.learning_rate)
        self.critic3_optimizer = Adam(self.critic3.parameters(), lr = self.learning_rate)

    def advantage_function(self, fromTimestep):
        advantage = 0

        critic_values = []
        for timestep in range(len(self.replay_buffer.buffer)):
            trajectory = self.replay_buffer.buffer[timestep]
            state = trajectory[0]

            # Using multiple critics
            critic1_value = self.critic.forward(torch.as_tensor(state))
            critic2_value = self.critic2.forward(torch.as_tensor(state))
            critic3_value = self.critic3.forward(torch.as_tensor(state))

            critic_values.append((critic1_value + critic2_value + critic3_value) / 3)

        for timestep in reversed(range(fromTimestep)):
            trajectory = self.replay_buffer.buffer[timestep]
            reward = trajectory[2]
            advantage += reward + (self.gamma * critic_values[timestep + 1]) - critic_values[timestep]

        return advantage

    def update_networks(self):
        batch_state, batch_action, batch_reward, batch_obs, batch_rtg = zip(*self.replay_buffer.buffer)
        
        actor_loss = self.calculate_actor_loss(batch_state, batch_action, batch_reward, batch_obs, batch_rtg)
        critic_loss = self.calculate_critic_loss(batch_state, batch_action, batch_reward, batch_obs, batch_rtg)

        # Update gradients
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        self.critic_optimizer.zero_grad()
        self.critic2_optimizer.zero_grad()
        self.critic3_optimizer.zero_grad()

        critic_loss.backward()

        self.critic_optimizer.step()
        self.critic2_optimizer.step()
        self.critic3_optimizer.step()

        return { 'actor_loss': actor_loss, 'critic_loss': critic_loss }