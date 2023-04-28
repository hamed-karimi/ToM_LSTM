import torch.nn as nn
import torch


class MentalNet(nn.Module):
    def __init__(self, env_size, states_num, layers_num):
        super(MentalNet, self).__init__()
        self.h_0 = None
        self.c_0 = None
        self.num_layers = layers_num
        self.lstm = nn.LSTM(input_size=env_size,
                            hidden_size=states_num,
                            num_layers=layers_num,
                            batch_first=True)
        self.hidden_size = states_num
        # self.fc1 = nn.Linear(traits_size+env_size+hidden_size,states_size)

    def forward(self, environment, reinitialize):
        # maybe we should save h_0 and c_0 for next predictions, as the sequence is basically infinite
        batch_size = environment.shape[0]
        if reinitialize:
            self.h_0 = torch.zeros((self.num_layers, batch_size, self.hidden_size), requires_grad=True)
            self.c_0 = torch.zeros((self.num_layers, batch_size, self.hidden_size), requires_grad=True)

        # combined = torch.cat((mental_trait, environment), dim=2)  #.unsqueeze(dim=1)  # batch_size * seq_len * features
        mental_states, (h_n, c_n) = self.lstm(environment, (self.h_0, self.c_0))

        hidden = (h_n, c_n)
        return mental_states, hidden
