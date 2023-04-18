import torch.nn as nn
import torch


class GoalNet(nn.Module):
    def __init__(self, traits_size, states_size, goal_num):
        super(GoalNet, self).__init__()
        self.fc1 = nn.Linear(traits_size + states_size, goal_num+1)  # +1 for staying

    def forward(self, traits_a, mental_states):
        combined = torch.cat((traits_a, mental_states.squeeze(dim=1)), 1)  # Squeeze the sequence length
        actions = self.fc1(combined)
        return actions
