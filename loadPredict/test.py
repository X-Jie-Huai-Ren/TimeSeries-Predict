


import torch
from torch import nn




class CNN(nn.Module):
    
    def __init__(self) -> None:
        super(CNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=6, out_channels=64, kernel_size=2)
        )


    def forward(self, x):

        return self.conv1(x)
    



if __name__ == '__main__':

    # generate the data
    data = torch.randn((1, 6, 24))

    model = CNN()

    output = model(data)

    print(output.shape)