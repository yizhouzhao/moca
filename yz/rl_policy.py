import torch
import torch.nn as nn
from torch.distributions.one_hot_categorical import OneHotCategorical

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

    def sample_action_with_prob(self, x):
        likelihood = self.forward(x)
        m = OneHotCategorical(likelihood)
        action = m.sample()
        return action, m.log_prob(action)


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim

        self.linear = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, s, a):
        x = torch.cat([s, a], dim = 1)
        x = self.linear(x)
        return x
    