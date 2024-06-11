import torch
import math
from torch import nn
from util.MultiHeadAttention import MultiHeadAttention
from util.PositionalEncoding import PositionalEncoding
def xavier_init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
    if type(m) == nn.RNN:
        for param in m._flat_weights_names:
            if "weight" in param:
                nn.init.xavier_uniform_(m._parameters[param])

class PositionWiseFFN(nn.Module):
    def __init__(self,ffn_num_inputs,ffn_num_hiddens,ffn_num_outputs):
        super(PositionWiseFFN, self).__init__()
        self.liner1 = nn.Linear(ffn_num_inputs,ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.liner2 = nn.Linear(ffn_num_hiddens,ffn_num_outputs)
    def forward(self,X):
        X = self.liner1(X)
        X = self.relu(X)
        X = self.liner2(X)
        return X
class AddNorm(nn.Module):
    def __init__(self,normalized_shape,dropout):
        super(AddNorm, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)
    def forward(self,X,Y):
        out = self.dropout(Y)+X  #似乎是把transform的某一时刻输出Y和输入X相加
                                #num_hiddens必须等于vocab_size
        out = self.ln(out)
        return out

class EncoderBlock(nn.Module):
    def __init__(self,key_size,query_size,value_size,num_hiddens,
                 norm_shape,ffn_num_inputs,ffn_num_hiddens,num_heads,
                 dropout,use_bias=False):
        super(EncoderBlock, self).__init__()
        self.attention = MultiHeadAttention(key_size,query_size,value_size,num_hiddens,num_heads,dropout,use_bias)
        #bz,qn,num_hiddens
        self.addnorm1 = AddNorm(norm_shape,dropout)
        self.ffn = PositionWiseFFN(ffn_num_inputs,ffn_num_hiddens,num_hiddens)
        #为啥最后一个是num_hiddens?因为维度相同才可以加
        self.addnorm2 = AddNorm(norm_shape,dropout)
    def forward(self,X,valid_lens):
        M_out = self.attention(X,X,X,valid_lens) #bz,num_queries,num_hiddens
        Y = self.addnorm1(X,M_out)  #bz,num_queries,num_hiddens
        ffn_out = self.ffn(Y) #bz,num_queries,num_hiddens
        Y = self.addnorm2(Y,ffn_out)
        return Y
class TransformerEncoder(nn.Module):
    def __init__(self,vocab_size,key_size=300,query_size=300,value_size=300,
                 num_hiddens=300,norm_shape=[300],ffn_num_inputs=300,ffn_num_hiddens=600,
                 num_heads=10,num_layers=1,dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.num_hiddens =num_hiddens
        self.embedding =nn.Linear(vocab_size,num_hiddens)
        self.pos_encoding = PositionalEncoding(num_hiddens,dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):  #都多少个block
            self.blks.add_module('block'+str(i),
            EncoderBlock(key_size,query_size,value_size,num_hiddens,norm_shape,ffn_num_inputs,
                         ffn_num_hiddens,num_heads,dropout))

    def forward(self,X):
        X = self.embedding(X)
        X = self.pos_encoding(X*math.sqrt(self.num_hiddens))
        self.attention_weights = [None]*len(self.blks) #每块都存
        for i ,blk in enumerate(self.blks):
            X =blk(X,None)
            self.attention_weights[i] = blk.attention.attention.attention_weights
        return X

class DecoderBlock(nn.Module):
    def __init__(self,key_size,query_size,value_size,num_hiddens,
                 norm_shape,ffn_num_inputs,ffn_num_hiddens,num_heads,
                 dropout,i):
        super(DecoderBlock, self).__init__()
        self.i = i
        self.attention1 = MultiHeadAttention(key_size,query_size,value_size,num_hiddens,num_heads,dropout)
        #bz,qn,num_hiddens
        self.addnorm1 = AddNorm(norm_shape,dropout)
        self.attention2 = MultiHeadAttention(key_size,query_size,value_size,num_hiddens,num_heads,dropout)
        self.addnorm2 = AddNorm(norm_shape,dropout)
        self.ffn = PositionWiseFFN(ffn_num_inputs,ffn_num_hiddens,num_hiddens)
        #为啥最后一个是num_hiddens?因为维度相同才可以加
        self.addnorm3 = AddNorm(norm_shape,dropout)
    def forward(self,X,state):
        enc_outputs,enc_valid_lens = state[0],state[1]
        if state[2][self.i] is None:    #train就是X
            key_values = X
        else:                           #推理就是concat，因为看不到所有，把之前的累积起来
            key_values = torch.cat((state[2][self.i],X),axis=1)
        state[2][self.i] = key_values
        if self.training:
            batch_size,num_steps,_ = X.shape
            dec_valid_lens = torch.arange(1,num_steps+1,device=X.device).repeat(batch_size,1)
            # 这里不明白可以看眼书
        else:
            dec_valid_lens = None   #predict 就不需要啦因为是一个一个过来的
        X2 = self.attention1(X,key_values,key_values,dec_valid_lens)
        Y = self.addnorm1(X,X2)
        Y2 = self.attention2(Y,enc_outputs,enc_outputs,enc_valid_lens)
        Z = self.addnorm2(Y,Y2)
        out =self.addnorm3(Z,self.ffn(Z))
        return out,state
class TransformerDecoder(nn.Module):
    def __init__(self,vocab_size,key_size=300,query_size=300,value_size=300,
                 num_hiddens=300,norm_shape=[300],ffn_num_inputs=300,ffn_num_hiddens=600,
                 num_heads=10,num_layers=1,dropout=0.1):
        super(TransformerDecoder, self).__init__()
        self.num_hiddens =num_hiddens
        self.num_layers =num_layers
        self.embedding = nn.Linear(vocab_size,num_hiddens)
        self.pos_encoding = PositionalEncoding(num_hiddens,dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module('block'+str(i),
                DecoderBlock(key_size,query_size,value_size,num_hiddens,norm_shape,
                             ffn_num_inputs,ffn_num_hiddens,num_heads,dropout,i))
        self.linear = nn.Linear(num_hiddens,vocab_size)

    def init_state(self,enc_outputs,enc_valid_lens):
        return [enc_outputs,enc_valid_lens,[None]*self.num_layers]
    def forward(self,X,state):
        X = self.embedding(X)
        X = self.pos_encoding(X*math.sqrt(self.num_hiddens))
        self._attention_weights = [[None]*len(self.blks) for _ in range(2)] #有两个注意力块
        for i,blk in enumerate(self.blks): #一层一层来
            X,state = blk(X,state)
            self._attention_weights[0][i] = blk.attention1.attention.attention_weights
            self._attention_weights[1][i] = blk.attention2.attention.attention_weights
        out = self.linear(X)
        out = out[:,-1,:]
        return out,state
    def attention_weights(self):
        return self._attention_weights
class EncoderDecoder(nn.Module):
    def __init__(self,encoder,decoder):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    def forward(self,X):
        enc_outputs = self.encoder(X)
        dec_state = self.decoder.init_state(enc_outputs,None)
        out,state =self.decoder(X,dec_state)
        return out
    
