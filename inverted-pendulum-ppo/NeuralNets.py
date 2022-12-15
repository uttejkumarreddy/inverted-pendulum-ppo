import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Normal

class ActorNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, n_layers):
        super(ActorNN, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.tanh = nn.Tanh()

        self.input_layer = nn.Linear(self.input_size, self.hidden_size)
        self.hidden_layer = nn.Linear(self.hidden_size, self.hidden_size)
        self.output_layer = nn.Linear(self.hidden_size, self.output_size)
    
    def forward(self, state):
        network = self.tanh(self.input_layer(state))
        for i in range(self.n_layers):
            network = self.tanh(self.hidden_layer(network))
        mean = self.output_layer(network)

        log_std = -0.5 * np.ones(self.output_size, dtype=np.float32)
        log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        log_std = torch.exp(log_std)
        
        return mean, log_std

# Task 5: Add FFN for Critic
class CriticNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, n_layers):
        super(CriticNN, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.tanh = nn.Tanh()

        self.input_layer = nn.Linear(self.input_size, self.hidden_size)
        self.hidden_layer = nn.Linear(self.hidden_size, self.hidden_size)
        self.output_layer = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, state):
        state = torch.as_tensor(state)

        network = self.tanh(self.input_layer(state))
        for i in range(self.n_layers):
            network = self.tanh(self.hidden_layer(network))
        network = self.output_layer(network)

        value = network.item()

        return value
        
class NormalModule(nn.Module):
    def __init__(self, inp, out, activation=nn.Tanh):
        super().__init__()
        self.m = nn.Linear(inp, out)
        log_std = -0.5 * np.ones(out, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.act1 = activation

    def forward(self, inputs):
        mout = self.m(inputs)
        vout = torch.exp(self.log_std)
        return mout, vout