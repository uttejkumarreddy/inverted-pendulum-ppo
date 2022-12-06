import gym
import torch
import numpy as np

from Modules import NormalModule
from ExperienceReplayBuffer import ExperienceReplayBuffer
from EpisodeRewardProcessingFunction import calculate_discounted_reward_to_go

from torch.distributions import Normal

env = gym.make('Pendulum-v1')

# sample hyperparameters
batch_size = 10000
epochs = 30
learning_rate = 1e-2
hidden_size = 8
n_layers = 2

# additional hyperparameters
gamma = 0.99

TRAINING_SIZE = 5

# Task 1: Environment interaction loop
# state = env.reset()

# Experience replay buffer
replayBuffer = ExperienceReplayBuffer(TRAINING_SIZE)

for episode in range(epochs):
    state = env.reset()

    # Task 3: Make episode reward processing function
    episodicReward = 0
    episodeRewards = []

    for _ in range(TRAINING_SIZE):
        # Task 1: Environment Interaction Loop
        # action = env.action_space.sample()

        # Task 2: Test experience replay buffer with random policy from gaussian distribution
        INPUT_SIZE = 3 # state dimensions
        OUTPUT_SIZE = 1

        normalModule = NormalModule(INPUT_SIZE, OUTPUT_SIZE)
        mean, std = normalModule.forward(torch.as_tensor(state))

        gaussianDist = Normal(mean, std)
        action = gaussianDist.sample().item()

        # Apply action
        obs, reward, done, info = env.step([action])

        # Task 2: Store trajectory in experience replay buffer
        experience = [state, action, reward, obs]
        replayBuffer.append(experience)

        env.render()
    
    # Task 3: Calculate episodic reward
    episodicReward = calculate_discounted_reward_to_go(replayBuffer.buffer, gamma)        