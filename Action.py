import torch.nn as nn
import torch.nn.functional as F


class ActionNet(nn.Module):
    def __init__(self, states_size):
        super(ActionNet, self).__init__()
        self.fc1 = nn.Linear(states_size, 12)  # +1 for staying
        self.fc2 = nn.Linear(12, 9)

    def forward(self, mental_states):
        x = F.relu(mental_states)
        x = self.fc1(x)
        actions = self.fc2(F.relu(x))
        return actions
