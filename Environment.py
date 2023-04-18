import torch.nn as nn
import torch.nn.functional as F
import Utilities
import numpy as np

def num_flat_features(x):  # This is a function we added for convenience to find out the number of features in a layer.
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
        num_features *= s
    return num_features


class EnvironmentNet(nn.Module):
    # let's assume that the environment is a 8x8 gridworld
    def __init__(self, goal_num, env_character_num, kernel_size: list):
        super(EnvironmentNet, self).__init__()
        self.utility = Utilities.Utilities()
        self.params = self.utility.get_params()
        self.max_pool_kernel_size = self.params.ENVIRONMENT_MAXPOOL_KERNEL_SIZE
        self.env_characteristic_num = env_character_num
        self.conv1 = nn.Conv2d(goal_num+1, 32, kernel_size[0])  # 1 input image channel, 8 output channels, 3x3 square
        # convolution kernel
        self.conv2 = nn.Conv2d(32, 16, kernel_size[1])  # 8 channels from the conv1 layer, 16 output channels,
        # 3x3 square convolution kernel
        self.fc1 = nn.Linear(16, self.env_characteristic_num)  # 4*4 from image
        # dimension

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(F.relu(self.conv2(x)), self.max_pool_kernel_size)
        x = x.view(-1, num_flat_features(x))
        x = self.fc1(x)
        return x
