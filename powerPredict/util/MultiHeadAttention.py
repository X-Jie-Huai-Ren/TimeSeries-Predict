import torch
from torch import nn
from util.attention import DotProductAttention

#多头注意力


def transpose_qkv(X,num_heads):
    X = X.reshape(X.shape[0],X.shape[1],num_heads,-1)#(bz,num,heads,num_hiddens/heads)
    X = X.permute(0,2,1,3) #1和2调个儿
    X = X.reshape(-1,X.shape[2],X.shape[3]) #bz*heads,num,numhiddens/heads
    return X
def transpose_output(X,num_heads):
    X = X.reshape(-1,num_heads,X.shape[1],X.shape[2])
    X = X.permute(0,2,1,3)
    X = X.reshape(X.shape[0],X.shape[1],-1)
    return X

class MultiHeadAttention(nn.Module):
    def __init__(self,key_size,query_size,value_size,num_hiddens,num_heads,drop_out,bias=False):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.attention = DotProductAttention(drop_out)
        self.W_q = nn.Linear(query_size,num_hiddens,bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self,queries,keys,values,valid_lens):
        queries = transpose_qkv(self.W_q(queries),self.num_heads)#(bz*heads,queries_num,num_hiddens/num_heads)
        keys = transpose_qkv(self.W_k(keys),self.num_heads)
        values = transpose_qkv(self.W_v(values),self.num_heads)
        if valid_lens is not None:
            valid_lens = torch.repeat_interleave(valid_lens,repeats=self.num_heads,dim=0)
        output = self.attention(queries,keys,values,valid_lens) #(bz*heads,queries_num,num_hiddens/num_heads)
        output_concat = transpose_output(output,self.num_heads)#(bz,querise,num_hiddens)
        return self.W_o(output_concat) #再乘个w_o