import gym
import torch
import numpy as np

from Modules import NormalModule
from ExperienceReplayBuffer import ExperienceReplayBuffer

from torch.distributions import Normal

env = gym.make('Pendulum-v1')

# sample hyperparameters
batch_size = 10000
epochs = 30
learning_rate = 1e-2
hidden_size = 8
n_layers = 2

TRAINING_SIZE = 100

# Environment interaction loop
state = env.reset()

# Experience replay buffer
replayBuffer = ExperienceReplayBuffer(TRAINING_SIZE)

for _ in range(TRAINING_SIZE):
    # Task 1: Environment Interaction Loop
    # action = env.action_space.sample()

    # Task 2: Test experience replay buffer with random policy from gaussian distribution
    INPUT_SIZE = 3 # state dimensions
    OUTPUT_SIZE = 1
    normalModule = NormalModule(INPUT_SIZE, OUTPUT_SIZE)
    stateTensor = torch.as_tensor(state)
    mean, std = normalModule.forward(stateTensor)

    gaussianDist = Normal(mean, std)
    action = gaussianDist.sample().item()

    # Apply action
    obs, reward, done, info = env.step([action])

    # Task 2: Store trajectory in experience replay buffer
    experience = [state, action, reward, obs]
    replayBuffer.append(experience)

    env.render()