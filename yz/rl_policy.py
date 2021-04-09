import torch
import torch.nn as nn

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, action_output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = action_output_dim

        self.linear = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_output_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.linear(x)
        likelihood = torch.softmax(x, dim = -1)
        return likelihood
    