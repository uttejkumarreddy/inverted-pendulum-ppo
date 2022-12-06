import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Normal

class ActorNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(ActorNN, self).__init__()

        self.input_size = input_size
        self.output_size = output_size

        self.input_layer = nn.Linear(input_size, 32)

        self.hidden_layer_1 = nn.Linear(32, 128)
        self.hidden_layer_2 = nn.Linear(128, 256)

        self.output_layer_mean = nn.Linear(256, self.output_size)
    
    def forward(self, state):
        network = nn.ReLU(self.input_layer(state))
        network = nn.ReLU(self.hidden_layer_1(network))
        network = nn.ReLU(self.hidden_layer_2(network))
        mean = self.output_layer_mean(network)

        log_std = -0.5 * np.ones(self.output_size, dtype=np.float32)
        log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        log_std = torch.exp(log_std)

        normalDistribution = Normal(mean, log_std)
        action = normalDistribution.sample().item()
        
        return action

        
