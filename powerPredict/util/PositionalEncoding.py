import torch
from torch import nn

#位置编码

class PositionalEncoding(nn.Module):
    def __init__(self,num_hiddens,dropout,max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.p = torch.zeros((1,max_len,num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        # i=max_lens,j=num_hiddens
        #(10000,1)/10000^
        self.p[:,:,0::2] = torch.sin(X)
        self.p[:,:,1::2] = torch.cos(X)

    def forward(self,X):
        X = X + self.p[:,:X.shape[1],:].to(X.device)
        return self.dropout(X)