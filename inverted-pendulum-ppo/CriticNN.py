import torch.nn as nn
import torch

# Task 5: Add FFN for Critic
class CriticNN(nn.Module):
    def __init__(self):
        super(CriticNN, self).__init__()

        self.IN_SIZE = 3
        self.OUT_SIZE = 1

        self.relu = nn.ReLU()

        self.input_layer = nn.Linear(self.IN_SIZE, 32)

        self.hidden_layer_1 = nn.Linear(32, 128)
        self.hidden_layer_2 = nn.Linear(128, 256)

        self.output_layer = nn.Linear(256, self.OUT_SIZE)

    def forward(self, state):
        state = torch.as_tensor(state)

        network = self.relu(self.input_layer(state))
        network = self.relu(self.hidden_layer_1(network))
        network = self.relu(self.hidden_layer_2(network))
        network = self.output_layer(network)

        value = network.item()

        return value