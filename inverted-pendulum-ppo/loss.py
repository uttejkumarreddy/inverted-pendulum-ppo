
import torch
from EpisodeRewardProcessingFunction import calculate_discounted_reward_to_go
# Policy Gradient Loss:    Loss = E[log(pi(a|s)) * (R_to_go)]

gamma=0.99

def policy_gradient_loss(gamma, traj, policy_function, **kwargs):

    loss = 0
    for trajectory in traj:
        for time, timestep in enumerate(trajectory):
            reward = calculate_discounted_reward_to_go(trajectory, time, gamma)
            s, a = timestep[0], timestep[1]
            loss += torch.log(policy_function(a, s)) * reward

    loss *= 1 / len(traj) * 1 / len(traj[0])

    return loss

def policy_gradient_loss_with_advantage_function(gamma, traj, policy_function, value_function, **kwargs):

    loss = 0
    for trajectory in traj:
        for t, timestep in enumerate(trajectory):
            A = policy_gradient_loss_with_advantage_function(trajectory, value_function, t, gamma)
            s, a = timestep[0], timestep[1]
            loss += torch.log(policy_function(a, s)) * A
    loss *= 1 / len(traj) * 1 / len(traj[0])

    return loss

def surrogate_objective_loss(gamma, traj, ratio_function, value_function, **kwargs):

    loss = 0
    for trajectory in traj:
        for t, timestep in enumerate(trajectory):
            s, a = timestep[0], timestep[1]
            loss += ratio_function(a, s) * policy_gradient_loss_with_advantage_function(trajectory, value_function, gamma, t)
    loss *= 1 / len(traj) * 1 / len(traj[0])

    return loss

def clip_func(x, A):
    eps = 0.3 #try 0.2
    if A > 0:
        return torch.clip(x, x, 1 + eps)
    else:
        return torch.clip(x, 1 - eps, x)


def surrogate_objective_loss_clipped(gamma, traj, ratio_function, value_function, clip_value, **kwargs):
    loss = 0
    for trajectory in traj:
        for t, timestep in enumerate(trajectory):
            s, a = timestep[0], timestep[1]
            ratio = ratio_function(a, s)
            A = policy_gradient_loss_with_advantage_function(trajectory, value_function, gamma, t)
            l1 = clip_func(ratio, A, clip_value) * A
            l2 = ratio * A
            loss += torch.minimum(l1,l2)

    loss *= 1 / len(traj) * 1 / len(traj[0])

    return loss




