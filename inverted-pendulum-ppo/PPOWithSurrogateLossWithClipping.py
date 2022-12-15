import torch
from PPO import BasePPOAgent

class PPOWithSurrogateLossWithClipping(BasePPOAgent):
    def __init__(self):
        super(PPOWithSurrogateLossWithClipping, self).__init__()
        self.clip_value = 0.2

    def calculate_actor_loss(self, batch_state, batch_action, batch_reward, batch_obs, batch_rtg):
        length_trajectories = len(batch_state)

        actor_loss = 0
        for i in range(length_trajectories):
            state = batch_state[i]
            action = batch_action[i]

            action_ratio = self.action_ratio(state, action)
            advantage = self.advantage_function(i)

            surrogate1 = None
            if advantage > 0:
                surrogate1 = torch.clip(action_ratio, torch.as_tensor(action_ratio), 1 + self.clip_value)
            else:
                surrogate1 = torch.clip(action_ratio, torch.as_tensor(1 - self.clip_value), action_ratio) 

            surrogate2 = action_ratio * advantage
            actor_loss += torch.min(surrogate1, surrogate2)

        return actor_loss