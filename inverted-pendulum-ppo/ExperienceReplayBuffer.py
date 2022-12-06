from collections import deque
import numpy as np

class ExperienceReplayBuffer():
    def __init__(self, size):
        self.size = size
        self.buffer = deque(maxlen = size)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        randomIndices = np.random.choice(
            np.arange(len(self.buffer)),
            size = batch_size,
            replace = False
            )

        samples = []
        for index in randomIndices:
            sample = self.buffer[index]
            samples.append(sample)
        
        return samples
    
    def clear(self):
        self.buffer.clear()