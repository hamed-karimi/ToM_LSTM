import torch.nn as nn
import torch


class MentalNet(nn.Module):
    def __init__(self, env_size, traits_num, states_num, layers_num):
        super(MentalNet, self).__init__()
        self.h_0 = None
        self.c_0 = None
        self.num_layers = layers_num
        self.lstm = nn.LSTM(input_size=env_size + traits_num,
                            hidden_size=states_num,
                            num_layers=layers_num,
                            batch_first=True)
        self.hidden_size = states_num
        # self.fc1 = nn.Linear(traits_size+env_size+hidden_size,states_size)

    def forward(self, mental_trait, environment, reinitialize):
        # maybe we should save h_0 and c_0 for next predictions, as the sequence is basically infinite
        batch_size = mental_trait.shape[0]
        if reinitialize:
            self.h_0 = torch.zeros((self.num_layers, batch_size, self.hidden_size), requires_grad=True)
            self.c_0 = torch.zeros((self.num_layers, batch_size, self.hidden_size), requires_grad=True)
        # else:
        #     self.h_0 = self.h_0.detach()
        #     self.c_0 = self.c_0.detach()
        combined = torch.cat((mental_trait, environment), 1).unsqueeze(dim=1)  # Seq_len * batch_size * features
        mental_states, (h_n, c_n) = self.lstm(combined,
                                              (self.h_0.requires_grad_(), self.c_0.requires_grad_()))
        self.h_0, self.c_0 = h_n.detach(), c_n.detach()

        hidden = (h_n, c_n)
        return mental_states, hidden
