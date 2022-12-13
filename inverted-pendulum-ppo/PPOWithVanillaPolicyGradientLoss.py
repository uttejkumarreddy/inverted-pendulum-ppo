import torch
from PPO import BasePPOAgent

class PPOAgentWithVanillaPolicyGradientLoss(BasePPOAgent):
    def __init__(self):
        super(PPOAgentWithVanillaPolicyGradientLoss, self).__init__()

    def actor_loss(self):
        loss = 0

        for timestep in range(len(self.replay_buffer.buffer)):
            trajectory = self.replay_buffer.buffer[timestep]

            # As mentioned in README.md, using rewards-to-go to calculate loss instead of advantage
            rewardToGo = self.calculate_reward_to_go(timestep)
            loss += torch.log(
                self.get_probability_of_action_in_state(
                    trajectory[0],
                    trajectory[1]
                )) * rewardToGo

        return loss
