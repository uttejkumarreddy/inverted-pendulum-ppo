import torch
from PPO import BasePPOAgent

class PPOWithSurrogateLossWithClipping(BasePPOAgent):
    def __init__(self):
        super(PPOWithSurrogateLossWithClipping, self).__init__()
        self.clip_value = 0.2

    def update_networks(self):
        batch_state, batch_action, batch_reward, batch_obs, batch_rtg = zip(*self.replay_buffer.buffer)
        length_trajectories = len(batch_state)

        # Actor loss
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

        # Critic losses
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