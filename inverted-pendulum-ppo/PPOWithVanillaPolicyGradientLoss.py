import torch
from PPO import BasePPOAgent

class PPOAgentWithVanillaPolicyGradientLoss(BasePPOAgent):
    def __init__(self):
        super(PPOAgentWithVanillaPolicyGradientLoss, self).__init__()

    def actor_loss(self):
        loss = 0
        for timestep, trajectory in self.replay_buffer.buffer:
            rewardToGo = self.calculate_reward_to_go(timestep)
            loss += torch.log(
                self.get_probability_of_action_in_state(
                    trajectory[0],
                    trajectory[1]
                )) * rewardToGo
        return loss

if __name__ == 'main':
    print('uttej')
    agent = PPOAgentWithVanillaPolicyGradientLoss()
    agent.train()
