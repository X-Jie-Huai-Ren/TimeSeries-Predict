"""
LSTM Model for load prediction

* @auther: xuan
* @email: 1920425406@qq.com
* @date: 2024-05-23
"""

import torch
from torch import nn
from torch.nn import functional as F


class LSTMModel(nn.Module):
    """"""
    def __init__(self, input_size, hidden_size, num_layers) -> None:
        super(LSTMModel, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.linear1 = nn.Linear(in_features=24*hidden_size, out_features=64)
        self.linear2 = nn.Linear(in_features=64, out_features=1)


    def forward(self, x):

        h0 = torch.zeros(self.num_layers, x.size(1), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(1), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
    
        out = out.permute(1, 0, 2)
        out = torch.flatten(out, start_dim=1, end_dim=-1)

        out = F.relu(self.linear1(out))

        output = self.linear2(out)

        return output
    



if __name__ == '__main__':

    # generate the data
    data = torch.randn((24, 2, 5))

    model = LSTMModel(input_size=5, hidden_size=64, num_layers=1)

    output = model(data)


    print(output.shape)

