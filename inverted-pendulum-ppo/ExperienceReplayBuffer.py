from collections import deque

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
    
    def clear(self):
        self.buffer.clear()