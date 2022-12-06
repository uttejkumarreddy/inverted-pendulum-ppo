# Task 3: Make episodic reward processing function
def calculate_discounted_reward_to_go(trajectories, gamma):
    episodicReward = 0

    for timestep, trajectory in enumerate(trajectories):
        reward = trajectory[2]
        episodicReward += reward * (gamma ** timestep)

    return episodicReward

