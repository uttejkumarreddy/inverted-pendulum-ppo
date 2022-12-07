import torch
from PPO import BasePPOAgent

class PPOWithSurrogateLossWithoutClipping(BasePPOAgent):
    def __init__(self):
        super(PPOWithSurrogateLossWithoutClipping, self).__init__()

    def actor_loss(self):
        loss = 0
        for timestep in range(len(self.replay_buffer.buffer)):
            trajectory = self.replay_buffer.buffer[timestep]

            state = trajectory[0]
            action = trajectory[1]

            action_ratio = self.action_ratio(state, action)
            advantage = self.advantage_function(timestep)

            loss += action_ratio * advantage

        return loss
