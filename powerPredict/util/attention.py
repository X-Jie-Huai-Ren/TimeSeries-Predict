import math
import torch
from torch import nn
from torch.nn import functional as F

#注意力

def sequence_mask(X,valid_len,value=0):
    maxlen=X.size(1)
    mask = torch.arange((maxlen),dtype=torch.float32,device=X.device)[None,:]<valid_len[:,None]
    X[~mask] = value
    return X

def sequence_mask(X,valid_len,value=0):
    maxlen=X.size(1)
    mask = torch.arange((maxlen),dtype=torch.float32,device=X.device)[None,:]<valid_len[:,None]
    X[~mask] = value
    return X

#遮蔽softmax
def masked_softmax(X,valid_lens):
    if valid_lens is None:
        return F.softmax(X,dim=-1)
    else:
        shape=X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens,shape[1]) #复制[2,3]=>[2,2,3,3]
        else:
            valid_lens = valid_lens.reshape(-1) #变成一维
        X = sequence_mask(X.reshape(-1, shape[-1]), valid_lens,value=-1e6) #把填充变成一个很小的数
        return F.softmax(X.reshape(shape),dim=-1)

# print(masked_softmax(torch.rand(2, 2, 4), torch.tensor([2, 3])))
# print(masked_softmax(torch.rand(2, 2, 4), torch.tensor([[1, 3], [2, 4]])))

# 加型平均
class AdditiveAttention(nn.Module):
    def __init__(self,key_size,query_size,num_hiddens,dropout):
        super(AdditiveAttention, self).__init__()
        self.W_k = nn.Linear(key_size,num_hiddens,bias=False)
        self.W_q = nn.Linear(query_size,num_hiddens,bias=False)
        self.W_v = nn.Linear(num_hiddens,1,bias=False)
        self.dropout = nn.Dropout(dropout)
    def forward(self,queries,keys,values,valid_lens):
        queries = self.W_q(queries) #(batch_size,num_query,query_size)=>(batch_size,num_query,num_hiddens)
        keys = self.W_k(keys) #(batch_size,num_keys,key_size)=>(batch_size,num_keys,num_hiddens)
        features = queries.unsqueeze(2)+keys.unsqueeze(1) #(batch_size,num_querry,num_keys,num_hiddens)
        features = torch.tanh(features)
        scores = self.W_v(features).squeeze(-1) #(batch_size,num_query,num_key)
        self.attention_weights = masked_softmax(scores,valid_lens) #softmax
        return torch.bmm(self.dropout(self.attention_weights),values) #(batch_size,num_query,value_size)

#当query和key最后一维一样时
class DotProductAttention(nn.Module):
    def __init__(self,dropout):
        super(DotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)
    def forward(self,queries,keys,values,valid_lens=None):
        d=queries.shape[-1]
        scores = torch.bmm(queries,keys.transpose(1,2))/math.sqrt(d)
        self.attention_weights = masked_softmax(scores,valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)

