import gym
import torch
import numpy as np

from NormalModule import NormalModule
from ExperienceReplayBuffer import ExperienceReplayBuffer
from ActorNN import ActorNN
from CriticNN import CriticNN

from torch.distributions import Normal
from torch.optim import Adam

class BasePPOAgent:
    def __init__(self):
        # Environment
        self.env = gym.make('Pendulum-v1')
        
        self.state = self.env.reset()

        # sample hyperparameters
        self.batch_size = 10000
        self.epochs = 30
        self.learning_rate = 1e-2
        self.hidden_size = 8
        self.n_layers = 2

        # additional hyperparameters
        self.gamma = 0.99
        self.training_size = 5
        
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

    def train(self):
        for episode in range(self.epochs):
            state = self.env.reset()
            episodicReward = 0
            episodicLosses = []

            for step in range(self.training_size):
                # Task 1: Environment Interaction Loop
                # action = env.action_space.sample()

                # Task 2: Test experience replay buffer with random policy from gaussian distribution
                # normalModule = NormalModule(self.input_size, self.output_size)
                # mean, std = normalModule.forward(torch.as_tensor(state))
                # gaussianDist = Normal(mean, std)
                # action = gaussianDist.sample().item()

                mean, std = self.actor.forward(state)
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
                episodicLosses.append(total_loss)
                
                # Update gradients
                self.actor_optimizer.zero_grad()
                total_loss.backward()
                self.actor_optimizer.step()
                
                self.critic_optimizer.zero_grad()
                total_loss.backward()
                self.critic_optimizer.step()

                self.env.render()
    
        # Task 3: Calculate episodic reward
        episodicReward = self.calculate_reward_to_go(0)

    # Task 3: Make episodic reward processing function
    def calculate_reward_to_go(self, fromTimestep):
        rewardToGo = 0
        for timestep, trajectory in enumerate(self.replay_buffer.buffer[fromTimestep:]):
            reward = trajectory[2]
            rewardToGo += reward * (self.gamma ** timestep)
        return rewardToGo

    def get_probability_of_action_in_state(self, state, action):
        mean, std = self.actor.forward(state)
        probability = 1 / (std * torch.sqrt(2 * np.pi)) * torch.exp(-(action - mean)**2 / (2 * std**2))
        return probability

    def actor_loss():
        # Implemented in calling functions
        pass

    def critic_loss(self):
        rewardsTrue = []
        rewardsToGo = []

        for timestep, trajectory in enumerate(self.replay_buffer.buffer):
            rewardsTrue.append(trajectory[2])
            rewardsToGo.append(self.calculate_reward_to_go(timestep))

        mse = (rewardsTrue - rewardsToGo).pow(2).mean()
        return mse
    



        
