import torch
from PPO import BasePPOAgent

class PPOWithGeneralizedAdvantageLoss(BasePPOAgent):
    def __init__(self):
        super(PPOWithGeneralizedAdvantageLoss, self).__init__()

    def actor_loss(self):
        loss = 0

        for timestep in range(len(self.replay_buffer.buffer)):
            advantage = self.advantage_function(timestep)

            trajectory = self.replay_buffer.buffer[timestep]
            loss += torch.log(
                self.get_probability_of_action_in_state(
                    trajectory[0],
                    trajectory[1]
                )) * advantage

        return loss

