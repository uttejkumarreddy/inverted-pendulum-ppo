import gym
from PPO import PPOAgent
import time
import numpy as np
import matplotlib.pyplot as plt

from loss import surrogate_objective_loss

env = gym.make('Pendulum-v1')
loss_function = surrogate_objective_loss
gamma = .99
total_timesteps = 100000
bsize = 10
traj_len = 100
train_iterations = total_timesteps // (bsize * traj_len)
log_interval = train_iterations // 2

agent = PPOAgent(input_size=3, output_size=1, log_std=0.01, num_trajectories=10, trajectory_length=100, learning_rate=1e-3)

agent.train(env=env, max_iterations=train_iterations, log_interval=log_interval)


iterations = np.arange(train_iterations)
plt.plot(iterations, agent.rewards)
plt.xlabel('Iteration')
plt.ylabel('Reward to Go')
plt.title('Reward to Go vs. Iteration')
plt.show()


plt.plot(iterations, agent.losses)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss vs. Iteration')
plt.show()

