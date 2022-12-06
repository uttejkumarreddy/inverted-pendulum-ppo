import gym

env = gym.make('Pendulum-v1')

# sample hyperparameters
batch_size = 10000
epochs = 30
learning_rate = 1e-2
hidden_size = 8
n_layers = 2

# Environment interaction loop
obs = env.reset()

TRAINING_SIZE = 100
for _ in range(TRAINING_SIZE):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    env.render()