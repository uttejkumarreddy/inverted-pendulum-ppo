import torch

from PPO import BasePPOAgent
from CriticNN import CriticNN

class EnsembleCriticsPPOAgent(BasePPOAgent):
    def __init__(self):
        super(EnsembleCriticsPPOAgent, self).__init__()

        self.critic2 = CriticNN(self.input_size, self.output_size, 8, self.n_layers)
        self.critic3 = CriticNN(self.input_size, self.output_size, 32, self.n_layers)

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
        pass