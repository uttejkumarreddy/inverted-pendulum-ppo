import torch
from PPO import BasePPOAgent
import copy

class PPOAgentWithVanillaPolicyGradientLoss(BasePPOAgent):
    def __init__(self):
        super(PPOAgentWithVanillaPolicyGradientLoss, self).__init__()

    def calculate_actor_loss(self, batch_state, batch_action, batch_reward, batch_obs, batch_rtg):
        length_trajectories = len(batch_state)

        # Actor loss
        # From README, gradient ascent steps are done with the average of the gradient of log-likelihood over a trajectory weight by rewards-to-go from each state
        actor_loss = 0
        for i in range(length_trajectories):
            state = batch_state[i]
            action = batch_action[i]
            rtg = batch_rtg[i]
            actor_loss += torch.log(self.get_probability_of_action_in_state(state, action)) * rtg
        actor_loss = (actor_loss / length_trajectories)

        return actor_loss
