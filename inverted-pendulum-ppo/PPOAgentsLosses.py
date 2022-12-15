import torch
from BasePPOAgent import BasePPOAgent

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



