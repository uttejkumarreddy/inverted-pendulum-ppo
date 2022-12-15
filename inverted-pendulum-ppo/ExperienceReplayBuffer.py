from collections import deque
import numpy as np

class ExperienceReplayBuffer():
    def __init__(self, size):
        self.size = size
        self.buffer = deque(maxlen = size)

    def append(self, trajectory):
        self.buffer.append(trajectory) # state, action, reward, obs

    def append_rtgs(self):
      batch_state, batch_action, batch_reward, batch_obs = zip(*self.buffer)
      for count, trajectory in enumerate(self.buffer):
        rtg = 0
        for i in range(len(batch_reward[count:])):
            rtg += 0.99 ** i * batch_reward[i]
        trajectory.append(rtg)

    def append_advntgs(self):
        batch_state, batch_action, batch_reward, batch_obs, batch_rtgs = zip(*self.buffer)
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
    
    def clear(self):
        self.buffer.clear()