import torch
import torch.nn as nn
from torch.distributions import Categorical

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

    def sample_prob(self, x):
        likelihood = self.forward(x)
        m = Categorical(likelihood)
        action = m.sample()
        return action, m.log_prob(action)


class QNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.linear = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        x = self.linear(x)
        return x
    