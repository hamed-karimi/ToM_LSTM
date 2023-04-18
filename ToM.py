import torch
import torch.nn as nn
import torch.nn.functional as F
import ObjectFactory
import Utilities
from Traits import TraitsNet
from Environment import EnvironmentNet
from Mental import MentalNet
from Goal import GoalNet


class ToMNet(nn.Module):
    # this is meant to operate as the meta-controller
    def __init__(self):
        super(ToMNet, self).__init__()
        self.batch_size = 1
        self.utility = Utilities.Utilities()
        self.params = self.utility.get_params()
        factory = ObjectFactory.ObjectFactory(utility=self.utility)
        self.mental_traits = factory.get_traits_net()
        self.action_traits = factory.get_traits_net()
        self.environment_net = factory.get_environment_net()
        self.mental_net = factory.get_mental_net()
        self.goal_net = factory.get_goal_net()
        self.softmax = nn.LogSoftmax(dim=1)
        # self.hidden_size = states_size
        self.fc_action = nn.Linear(self.params.GOAL_NUM + 1 +  # +1 for staying
                                   self.params.ENVIRONMENT_CHARACTERISTICS_NUM +
                                   self.params.TRAITS_NUM, 9)

    def forward(self, environment: torch.Tensor, agent_appearance, reinitialize_mental): # environment is a list of tensors, each an episode
        mental_trait = self.mental_traits(agent_appearance)
        action_trait = self.action_traits(agent_appearance)
        env_representation = self.environment_net(environment)

        # mental_state shape: [batch_size, seq_len, #features]
        mental_states, hidden = self.mental_net(mental_trait, env_representation, reinitialize_mental)
        goals = self.goal_net(action_trait, mental_states)
        actions = self.fc_action(torch.cat([
            F.relu(goals),
            F.relu(env_representation),
            F.relu(action_trait)
        ], dim=1))
        goals_prob = self.softmax(goals)
        actions_prob = self.softmax(actions)
        return goals, goals_prob, actions, actions_prob




    # def goal_grid_representation(self, goal_index):
    #     grid = torch.zeros((self.params.HEIGHT, self.params.WIDTH))

