import torch
from PPO import BasePPOAgent

class PPOWithGeneralizedAdvantageLoss(BasePPOAgent):
    def __init__(self):
        super(PPOWithGeneralizedAdvantageLoss, self).__init__()

    def update_networks(self):
        batch_state, batch_action, batch_reward, batch_obs, batch_rtg = zip(*self.replay_buffer.buffer)
        length_trajectories = len(batch_state)

        # Actor loss
        # From README, gradient ascent steps are done with the average of the gradient of log-likelihood over a trajectory weight by rewards-to-go from each state
        actor_loss = 0
        for i in range(length_trajectories):
            advantage = self.advantage_function(i)

            state = batch_state[i]
            action = batch_action[i]
            
            actor_loss += torch.log(self.get_probability_of_action_in_state(state, action)) * advantage

        # Calculate losses
        critic_loss = self.critic_loss(batch_reward, batch_rtg)
        critic_loss = torch.as_tensor([critic_loss])
        critic_loss.requires_grad_()

        # Update gradients
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        return { 'actor_loss': actor_loss, 'critic_loss': critic_loss }
