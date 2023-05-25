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
        self.mental_states = None
        self.utility = Utilities.Utilities()
        self.params = self.utility.get_params()
        factory = ObjectFactory.ObjectFactory(utility=self.utility)
        self.environment_net = factory.get_environment_net()
        self.mental_net = factory.get_mental_net()
        self.goal_net = factory.get_goal_net()
        self.softmax = nn.LogSoftmax(dim=1)
        self.action_net = factory.get_action_net()

    def forward(self, environment: torch.Tensor, reinitialize_mental, recalculate_mental=True,
                predefined_goals=torch.empty(0), goal_reached=torch.empty(0)):

        # environment is a tensor containing the steps of the whole episode
        if recalculate_mental:
            return self.forward_from_new_mental(environment, reinitialize_mental)
        else:
            return self.forward_action_layer(predefined_goals, goal_reached)

    def forward_from_new_mental(self, environment, reinitialize_mental):
        episode_len = environment.shape[1]
        env_representation_seq = []
        for step in range(episode_len):
            env_representation_seq.append(self.environment_net(environment[:, step, :, :, :]))

        # mental_state shape: [batch_size, seq_len, #features]
        self.mental_states, hidden = self.mental_net(torch.stack(env_representation_seq, dim=1),
                                                     reinitialize_mental)
        # mental_state : 256, 100, 4
        goal_seq, goal_prob_seq = [], []
        action_seq, action_prob_seq = [], []
        for step in range(episode_len):
            step_goal = self.goal_net(self.mental_states[:, step, :])
            goal_seq.append(step_goal)
            goal_prob_seq.append(self.softmax(step_goal))

            step_action = self.action_net(self.mental_states[:, step, :])
            action_seq.append(step_action)
            action_prob_seq.append(self.softmax(step_action))

        goals = torch.stack(goal_seq, dim=1)
        goals_prob = torch.stack(goal_prob_seq, dim=1)

        actions = torch.stack(action_seq, dim=1)
        actions_prob = torch.stack(action_prob_seq, dim=1)

        return goals, goals_prob, actions, actions_prob

    def forward_action_layer(self, predefined_goals, goal_reached):
        reached_goals = predefined_goals[goal_reached]
        binary_goals = torch.zeros(reached_goals.shape[0], self.params.GOAL_NUM + 1)
        binary_goals.index_put_((torch.arange(reached_goals.shape[0]), reached_goals), torch.ones(reached_goals.shape[0]))
        actions = self.fc_action(torch.cat([
            F.relu(binary_goals),
            F.relu(self.mental_states[goal_reached, :])
        ], dim=1))
        actions = self.softmax(actions)
        return actions
