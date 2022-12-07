import torch
import torch.nn as nn

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