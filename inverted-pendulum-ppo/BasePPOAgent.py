import gym
import torch
import numpy as np
import copy
import matplotlib.pyplot as plt

from NeuralNets import ActorNN, CriticNN, NormalModule
from ExperienceReplayBuffer import ExperienceReplayBuffer

from torch.distributions import Normal
from torch.optim import Adam

from collections import deque

class BasePPOAgent:
    def __init__(self, actor, critic):
        # Environment
        self.env = gym.make('Pendulum-v1')
        
        self.state = self.reset_env()

        # sample hyperparameters
        self.learning_rate = 3e-4
        self.hidden_size = 64
        self.n_layers = 2

        # additional hyperparameters
        self.gamma = 0.99
        self.timesteps_total = 120000
        self.timesteps_episode = 200
        self.timesteps_to_update = 400
        
        self.input_size = 3
        self.output_size = 1

        # Experience replay buffer
        self.replay_buffer = ExperienceReplayBuffer(self.timesteps_to_update)

        # Actor and critic networks
        self.actor = actor
        self.critic = critic

        # Actor network copy for Surrogative objective
        self.actor_old = actor

        # Actor and critic optimizers
        self.actor_optimizer = Adam(self.actor.parameters(), lr = self.learning_rate)
        self.critic_optimizer = Adam(self.critic.parameters(), lr = self.learning_rate)

        # Capture rewards and losses
        self.episodic_losses = []
        self.episodic_rewards = []

    def train(self):
        iter_timesteps_taken = 0
        update_count = 0

        while iter_timesteps_taken < self.timesteps_total:

            state = self.reset_env()
            episode_reward = 0
            losses = { 'actor_loss': 0, 'critic_loss': 0 }

            iter_timesteps_episode = 0
            while iter_timesteps_episode < self.timesteps_episode:
                # Task 1: Environment Interaction Loop
                # action = env.action_space.sample()

                # Task 2: Test experience replay buffer with random policy from gaussian distribution
                # normalModule = NormalModule(self.input_size, self.output_size)
                # mean, std = normalModule.forward(torch.as_tensor(state))
                # gaussianDist = Normal(mean, std)
                # action = gaussianDist.sample().item()

                iter_timesteps_taken += 1

                mean, std = self.actor.forward(torch.as_tensor(state))
                normalDistribution = Normal(mean, std)
                action = normalDistribution.sample().item()

                # Apply action
                obs, reward, done, info = self.apply_action(action)
                episode_reward += reward

                # Task 2: Store trajectory in experience replay buffer
                trajectory = [state, action, reward, obs]
                self.replay_buffer.append(trajectory)

                # Update actor and critic networks 
                if iter_timesteps_taken % self.timesteps_to_update == 0:
                    self.replay_buffer.append_rtgs()
                    losses = self.update_networks()
                    update_count += 1

                    print ('Update', update_count, 'Reward', episode_reward, 'Actor loss', losses['actor_loss'], 'Critic loss', losses['critic_loss'])

                    self.replay_buffer.clear()

                if done:
                    break

                state = obs

            self.episodic_rewards.append(episode_reward)
            self.episodic_losses.append(losses['actor_loss'] + losses['critic_loss'])

    # Task 3: Make episodic reward processing function
    def calculate_reward_to_go(self, fromTimestep):
        rewardToGo = 0
        for timestep in range(fromTimestep, len(self.replay_buffer.buffer)):
            trajectory = self.replay_buffer.buffer[timestep]
            reward = trajectory[2]
            rewardToGo += reward * (self.gamma ** timestep)
        return rewardToGo

    # Task 4: Vanilla Policy Gradient Agent
    def get_probability_of_action_in_state(self, state, action, usePreviousActor = False):
        mean, std = None, None
        if usePreviousActor:
            mean, std = self.actor_old.forward(torch.as_tensor(state))
        else:
            mean, std = self.actor.forward(torch.as_tensor(state))
        a = std * torch.sqrt(2 * torch.as_tensor(np.pi))
        b = torch.exp(-(action - mean)**2 / (2 * std**2))
        probability = 1 / (a * b)
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

    # Task 7: Surrogate Objective Functions
    def action_ratio(self, state, action):
        probability_of_action_in_current_actor = self.get_probability_of_action_in_state(state, action)
        probability_of_action_in_old_actor = self.get_probability_of_action_in_state(state, action, True)
        return (probability_of_action_in_current_actor / probability_of_action_in_old_actor)

    def surrogate_loss_function(self):
        pass

    # Actor Functions
    def actor_loss(self):
        # Implemented in calling functions
        pass

    # Critic Functions
    def critic_loss(self, rewards, rtgs):
        if rewards:
            rewards = np.array(rewards)
            rtgs = np.array(rtgs)
            mse = ((rewards - rtgs) ** 2).mean()
            return mse

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

    # Plot functions
    def plot_episodic_losses(self):
        plt.plot(
            np.arange(len(self.episodic_losses)),
            torch.tensor(self.episodic_losses).detach().numpy()
        )
        plt.xlabel('Iterations')
        plt.xlabel('Losses')
        plt.show()

    def plot_episodic_rewards(self):
        plt.plot(
            np.arange(len(self.episodic_rewards)),
            torch.tensor(self.episodic_rewards).detach().numpy()
        )
        plt.xlabel('Iterations')
        plt.xlabel('Rewards')
        plt.show()

    # Gradient update functions
    def calculate_actor_loss(self, batch_state, batch_action, batch_reward, batch_obs, batch_rtg):
        pass

    def calculate_critic_loss(self, batch_state, batch_action, batch_reward, batch_obs, batch_rtg):
        critic_loss = self.critic_loss(batch_reward, batch_rtg)
        critic_loss = torch.as_tensor([critic_loss])
        critic_loss.requires_grad_()
        return critic_loss

    def update_networks(self):
        batch_state, batch_action, batch_reward, batch_obs, batch_rtg = zip(*self.replay_buffer.buffer)
        
        actor_loss = self.calculate_actor_loss(batch_state, batch_action, batch_reward, batch_obs, batch_rtg)
        critic_loss = self.calculate_critic_loss(batch_state, batch_action, batch_reward, batch_obs, batch_rtg)

        # Update gradients
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        return { 'actor_loss': actor_loss, 'critic_loss': critic_loss }

    # Apply action and get observation from environment
    def reset_env(self):
        state = self.env.reset()
        return state

    def apply_action(self, action):
        obs, reward, done, info = self.env.step([action])
        return obs, reward, done, info




