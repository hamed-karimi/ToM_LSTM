import torch
from Environment import EnvironmentNet
from Traits import TraitsNet
from Mental import MentalNet
from Goal import GoalNet
from ToM import ToMNet
import torch.nn.init as init


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.orthogonal_(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal_(m.weight.data, gain=1)
    elif classname.find('BatchNorm') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


class ObjectFactory:
    def __init__(self, utility):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.environment_net = None
        self.mental_net = None
        self.trait_nest = None
        self.tom_net = None
        self.params = utility.get_params()

    def get_environment_net(self):
        environment_net = EnvironmentNet(goal_num=self.params.GOAL_NUM,
                                         env_character_num=self.params.ENVIRONMENT_CHARACTERISTICS_NUM,
                                         kernel_size=self.params.ENVIRONMENT_KERNEL_SIZE_LIST).to(self.device)
        environment_net.apply(weights_init_orthogonal)
        return environment_net

    def get_traits_net(self):
        trait_net = TraitsNet(agent_size=self.params.AGENT_SIZE,
                              traits_num=self.params.TRAITS_NUM).to(self.device)
        trait_net.apply(weights_init_orthogonal)
        return trait_net

    def get_mental_net(self):
        mental_net = MentalNet(env_size=self.params.ENVIRONMENT_CHARACTERISTICS_NUM,
                               states_num=self.params.MENTAL_STATES_NUM,
                               layers_num=self.params.LSTM_LAYERS_NUM).to(self.device)
        mental_net.apply(weights_init_orthogonal)
        return mental_net

    def get_goal_net(self):
        goal_net = GoalNet(states_size=self.params.MENTAL_STATES_NUM,
                           goal_num=self.params.GOAL_NUM).to(self.device)
        goal_net.apply(weights_init_orthogonal)
        return goal_net

    def get_tom_net(self):
        self.tom_net = ToMNet().to(self.device)
        self.tom_net.apply(weights_init_orthogonal)
        return self.tom_net
