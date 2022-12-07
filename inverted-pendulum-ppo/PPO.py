import gym
import torch
import numpy as np
import itertools
import matplotlib.pyplot as plt

from NormalModule import NormalModule
from ExperienceReplayBuffer import ExperienceReplayBuffer
from ActorNN import ActorNN
from CriticNN import CriticNN

from torch.distributions import Normal
from torch.optim import Adam

from collections import deque

class BasePPOAgent:
    def __init__(self):
        # Environment
        self.env = gym.make('Pendulum-v1')
        
        self.state = self.env.reset()

        # sample hyperparameters
        self.batch_size = 10000
        self.epochs = 1
        self.learning_rate = 1e-3
        self.hidden_size = 8
        self.n_layers = 2

        # additional hyperparameters
        self.gamma = 0.99
        self.training_size = 10000
        
        self.input_size = 3
        self.output_size = 1

        # Experience replay buffer
        self.replay_buffer = ExperienceReplayBuffer(self.training_size)

        # Actor and critic networks
        self.actor = ActorNN(self.input_size, self.output_size, self.hidden_size, self.n_layers)
        self.critic = CriticNN(self.input_size, self.output_size, self.hidden_size, self.n_layers)

        # Actor and critic optimizers
        self.actor_optimizer = Adam(self.actor.parameters(), lr = self.learning_rate)
        self.critic_optimizer = Adam(self.actor.parameters(), lr = self.learning_rate)

        # Capture rewards and losses
        self.episodic_losses = []
        self.episodic_rewards = []

    def train(self):
        for episode in range(self.epochs):
            state = self.env.reset()
            self.episodic_losses = []
            self.episodic_rewards = []

            for timestep in range(self.training_size):
                # Task 1: Environment Interaction Loop
                # action = env.action_space.sample()

                # Task 2: Test experience replay buffer with random policy from gaussian distribution
                # normalModule = NormalModule(self.input_size, self.output_size)
                # mean, std = normalModule.forward(torch.as_tensor(state))
                # gaussianDist = Normal(mean, std)
                # action = gaussianDist.sample().item()

                mean, std = self.actor.forward(torch.as_tensor(state))
                normalDistribution = Normal(mean, std)
                action = normalDistribution.sample().item()

                # Apply action
                obs, reward, done, info = self.env.step([action])

                # Task 2: Store trajectory in experience replay buffer
                trajectory = [state, action, reward, obs]
                self.replay_buffer.append(trajectory)

                # Calculate losses
                actor_loss = self.actor_loss()
                critic_loss = self.critic_loss()

                # Calculate and store total loss
                total_loss = actor_loss - critic_loss
                self.episodic_losses.append(total_loss)

                self.episodic_rewards.append(
                    self.calculate_reward_to_go(timestep)
                )

                # Update gradients
                self.actor_optimizer.zero_grad()
                total_loss.backward()
                self.actor_optimizer.step()
                
                self.critic_optimizer.zero_grad()
                self.critic_optimizer.step()

                self.env.render()

                state = obs     

    # Task 3: Make episodic reward processing function
    def calculate_reward_to_go(self, fromTimestep):
        rewardToGo = 0
        for timestep in range(fromTimestep, len(self.replay_buffer.buffer)):
            trajectory = self.replay_buffer.buffer[timestep]
            reward = trajectory[2]
            rewardToGo += reward * (self.gamma ** timestep)
        return rewardToGo

    # Task 4: Vanilla Policy Gradient Agent
    def get_probability_of_action_in_state(self, state, action):
        mean, std = self.actor.forward(torch.as_tensor(state))
        a = std * torch.sqrt(2 * torch.as_tensor(np.pi))
        b = torch.exp(-(action - mean)**2 / (2 * std**2))
        probability = 1 / a * b
        return probability

    # Task 6: Generalized Advantage
    def advantage_function(self, fromTimestep):
        advantage = 0

        critic_values = []
        for timestep in range(len(self.replay_buffer.buffer)):
            trajectory = self.replay_buffer.buffer[timestep]
            state = trajectory[0]
            critic_value = self.get_critic_value(state)
            critic_values.append(critic_value)

        for timestep in reversed(range(fromTimestep)):
            trajectory = self.replay_buffer.buffer[timestep]
            reward = trajectory[2]
            advantage += reward + (self.gamma * critic_values[timestep + 1]) - critic_values[timestep]

        return advantage

    # Actor Functions
    def actor_loss():
        # Implemented in calling functions
        pass

    # Critic Functions
    def critic_loss(self):
        rewardsTrue = []
        rewardsToGo = []

        for timestep in range(len(self.replay_buffer.buffer)):
            trajectory = self.replay_buffer.buffer[timestep]
            rewardsTrue.append(trajectory[2])
            rewardsToGo.append(self.calculate_reward_to_go(timestep))

        rewardsTrue = np.array(rewardsTrue)
        rewardsToGo = np.array(rewardsToGo)
        mse = ((rewardsTrue - rewardsToGo) ** 2).mean()

        return mse

    def get_critic_value(self, state):
        return self.critic.forward(torch.as_tensor(state))

    def plot_episodic_losses(self):
        plt.plot(
            np.arange(self.training_size),
            torch.tensor(self.episodic_losses).detach().numpy()
        )
        plt.xlabel('Iterations')
        plt.xlabel('Losses')
        plt.show()

    def plot_episodic_rewards(self):
        plt.plot(
            np.arange(self.training_size),
            torch.tensor(self.episodic_rewards).detach().numpy()
        )
        plt.xlabel('Iterations')
        plt.xlabel('Reward to Go')
        plt.show()