"""
CNN model for load prediction

* @auther: xuan
* @email: 1920425406@qq.com
* @date: 2024-05-21
"""

import torch
from torch import nn
from torch.nn import functional as F


class Model(nn.Module):
    """
    Conv1d model
    """
    def __init__(self, window_size, in_features) -> None:
        super(Model, self).__init__()

        self.window_size = window_size
        self.in_features = in_features

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=self.in_features, out_channels=64, kernel_size=3),  # 24 - 3 + 1 = 22
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=1)  # 22 - 3 + 1 = 20
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3),  # 20 - 3 + 1 = 18
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=1)  # 18 - 3 + 1 = 16
        )
        self.linear1 = nn.Linear(128 * 16, 64)
        self.linear2 = nn.Linear(64, 1)

    def forward(self, x):

        x = self.conv2(self.conv1(x))

        x = torch.flatten(x, start_dim=1, end_dim=-1)

        x = F.relu(self.linear1(x))

        output = self.linear2(x)

        return output
    


if __name__ == '__main__':

    # generate the data
    data = torch.randn((2, 5, 24))

    model = Model(window_size=24, in_features=5)

    output = model(data)

    print(output.shape)
