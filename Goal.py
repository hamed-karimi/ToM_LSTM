import torch.nn as nn
import torch


class GoalNet(nn.Module):
    def __init__(self, states_size, goal_num):
        super(GoalNet, self).__init__()
        self.fc1 = nn.Linear(states_size, goal_num+1)  # +1 for staying

    def forward(self, mental_states):
        actions = self.fc1(mental_states)
        return actions
