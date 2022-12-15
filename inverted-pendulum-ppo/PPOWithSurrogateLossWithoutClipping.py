import torch
from PPO import BasePPOAgent

class PPOWithSurrogateLossWithoutClipping(BasePPOAgent):
    def __init__(self):
        super(PPOWithSurrogateLossWithoutClipping, self).__init__()

    def calculate_actor_loss(self, batch_state, batch_action, batch_reward, batch_obs, batch_rtg):
        length_trajectories = len(batch_state)

        # Actor loss
        actor_loss = 0
        for i in range(length_trajectories):
            state = batch_state[i]
            action = batch_action[i]
            action_ratio = self.action_ratio(state, action)
            advantage = self.advantage_function(i)
            actor_loss += action_ratio * advantage

        return actor_loss