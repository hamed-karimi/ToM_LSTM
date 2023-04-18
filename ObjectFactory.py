from copy import deepcopy
from Environment import EnvironmentNet
from Traits import TraitsNet
from Mental import MentalNet
from Goal import GoalNet
from ToM import ToMNet


class ObjectFactory:
    def __init__(self, utility):
        self.environment_net = None
        self.mental_net = None
        self.trait_nest = None
        self.tom_net = None
        self.params = utility.get_params()

    def get_environment_net(self):
        return EnvironmentNet(goal_num=self.params.GOAL_NUM,
                              env_character_num=self.params.ENVIRONMENT_CHARACTERISTICS_NUM,
                              kernel_size=self.params.ENVIRONMENT_KERNEL_SIZE_LIST)

    def get_traits_net(self):
        return TraitsNet(agent_size=self.params.AGENT_SIZE,
                         traits_num=self.params.TRAITS_NUM)

    def get_mental_net(self):
        return MentalNet(env_size=self.params.ENVIRONMENT_CHARACTERISTICS_NUM,
                         traits_num=self.params.TRAITS_NUM,
                         states_num=self.params.MENTAL_STATES_NUM,
                         layers_num=self.params.LSTM_LAYERS_NUM)

    def get_goal_net(self):
        return GoalNet(traits_size=self.params.TRAITS_NUM,
                       states_size=self.params.MENTAL_STATES_NUM,
                       goal_num=self.params.GOAL_NUM)

    def get_tom_net(self):
        return ToMNet()

