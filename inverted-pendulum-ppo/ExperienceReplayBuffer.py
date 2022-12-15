from collections import deque
import numpy as np

class ExperienceReplayBuffer():
    def __init__(self, size):
        self.size = size
        self.buffer = deque(maxlen = size)

    def append(self, trajectory):
        self.buffer.append(trajectory)

    def append_rtgs(self):
      for count, trajectory in enumerate(self.buffer):
        rtg = 0
        for timestep in range(count, len(self.buffer)):
            trajectory = self.buffer[timestep]
            reward = trajectory[2]
            rtg += reward * (0.99 ** timestep)
        self.buffer[count].append(rtg)
    
    def clear(self):
        self.buffer.clear()