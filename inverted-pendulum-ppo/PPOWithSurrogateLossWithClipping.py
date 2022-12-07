import torch
from PPO import BasePPOAgent

class PPOWithSurrogateLossWithClipping(BasePPOAgent):
    def __init__(self):
        super(PPOWithSurrogateLossWithClipping, self).__init__()
        self.clip_value = 0.2

    def actor_loss(self):
        loss = 0
        for timestep in range(len(self.replay_buffer.buffer)):
            trajectory = self.replay_buffer.buffer[timestep]

            state = trajectory[0]
            action = trajectory[1]

            action_ratio = self.action_ratio(state, action)
            advantage = self.advantage_function(timestep)

            surrogate1 = None
            if advantage > 0:
                surrogate1 = torch.clip(action_ratio, torch.as_tensor(action_ratio), 1 + self.clip_value)
            else:
                surrogate1 = torch.clip(action_ratio, torch.as_tensor(1 - self.clip_value), action_ratio) 

            surrogate2 = action_ratio * advantage

            loss += torch.min(surrogate1, surrogate2)

        return loss