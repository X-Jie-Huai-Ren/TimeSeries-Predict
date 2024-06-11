"""
CNN model for load prediction

* @auther: xuan
* @email: 1920425406@qq.com
* @date: 2024-06-04
"""

import torch
from torch import nn
from torch.nn import functional as F


class Model(nn.Module):
    """
    Conv1d model
    """
    def __init__(self, window_size, in_features) -> None:
        """
        
        """
        super(Model, self).__init__()

        self.window_size = window_size
        self.in_features = in_features

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=self.in_features, out_channels=256, kernel_size=3),  # window_size - 2
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3),  # window_size - 4
            nn.ReLU(),
        )
        self.linear1 = nn.Linear(512 * (window_size-4), 256)
        self.linear2 = nn.Linear(256, 96)

    def forward(self, x):

        # change the dimensions of features for statisfying the Conv1d
        x = x.permute(0, 2, 1)

        x = self.conv2(self.conv1(x))

        x = torch.flatten(x, start_dim=1, end_dim=-1)

        x = F.relu(self.linear1(x))

        output = self.linear2(x)

        return output
    


if __name__ == '__main__':

    # generate the data
    data = torch.randn((2, 96, 7))

    model = Model(window_size=7, in_features=96)

    output = model(data)

    print(output.shape)
