"""
RNN Model for Synthetic data

* @auther: xuan
* @email: 1920425406@qq.com
* @date: 2023-12-28
"""

import sys
sys.path.append('D:\Python_WorkSpace\DL\TimeSeries')

import torch
from torch import nn
from torch.utils.data import DataLoader
from dataset.synthetic_dataset import SyntheticDataset
import config

class Model(nn.Module):
    """
    the simple RNN model
    """
    def __init__(self, input_dim, output_dim=1) -> None:
        super(Model, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        # the number of hidden layer units
        self.hidden_units = 16   

        # RNN Layer
        self.rnn_model = nn.RNN(
                input_size=self.input_dim,
                hidden_size=self.hidden_units,
                num_layers=1
            )

        # output layer
        self.output_layer = nn.Linear(
            in_features=self.hidden_units,
            out_features=self.output_dim
        )


    def forward(self, x, h_s):
        """
        Params:
            x: the input features, shape(时间步数, batch_size, 特征维度)
            h_s: the hidden state, shape(1, batch_size, 隐藏单元个数)
        """

        rnn_out, hs = self.rnn_model(x, h_s)  # output shape(时间步数, batch_size, 隐藏单元个数)

        # 选择最后一个时间步输出作为linear层的输入
        output = self.output_layer(rnn_out[-1])

        return output, hs




if __name__ == '__main__':

    # load the dataset
    synthetic_train = SyntheticDataset()
    synthetic_test = SyntheticDataset(train=False)

    # dataloader
    train_loader = DataLoader(
        dataset=synthetic_train,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        pin_memory=True
    )
    test_loader = DataLoader(
        dataset=synthetic_test,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        pin_memory=True
    )

    rnn = Model(input_dim=1, output_dim=3)

    h0 = None

    for x, y in train_loader:

        x = x.T.unsqueeze(2) # input shape: (时间步数, batch_size, 特征维度) （20, 32, 1）

        output, hn = rnn(x, h0)   

        print(output[-1]) # output 隐藏层在各个时间步上计算并输出的隐藏状态, shape(时间步数, batch_size, 隐藏单元个数), (20, 32, 3)
        print(hn)     # hn 隐藏层在最后时间步的隐藏状态，有多层时，每一层的隐藏状态都会记录 shape(层数, batch_size, 隐藏单元个数), (1, 32, 3)        
        
        break  

    print(rnn.state_dict())