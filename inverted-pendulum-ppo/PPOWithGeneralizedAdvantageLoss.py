import torch
from PPO import BasePPOAgent

class PPOWithGeneralizedAdvantageLoss(BasePPOAgent):
    def __init__(self):
        super(PPOWithGeneralizedAdvantageLoss, self).__init__()

    def calculate_actor_loss(self, batch_state, batch_action, batch_reward, batch_obs, batch_rtg):
        length_trajectories = len(batch_state)

        # Actor loss
        # From README, gradient ascent steps are done with the average of the gradient of log-likelihood over a trajectory weight by rewards-to-go from each state
        actor_loss = 0
        for i in range(length_trajectories):
            advantage = self.advantage_function(i)

            state = batch_state[i]
            action = batch_action[i]
            
            actor_loss += torch.log(self.get_probability_of_action_in_state(state, action)) * advantage
        
        return actor_loss
