from torch import nn
import torch
class Seq2Seq(nn.Module):
    def __init__(self, embed_size, num_hiddens):
        super(Seq2Seq, self).__init__()
        self.encode = nn.RNN(embed_size, num_hiddens)
        self.decode = nn.RNN(num_hiddens*2, embed_size)

    def forward(self,x):
        out_encode,state_encode = self.encode(x)   

        context = state_encode[-1].repeat(out_encode.shape[0],1,1)
        context_concat = torch.cat((context,out_encode),dim=2)
        out,_ = self.decode(context_concat)
        out = out[:,-1,:]
        return out      #(32,96)
    

if __name__ == '__main__':

    # generate the data
    data = torch.randn((2, 7, 96))

    model = Seq2Seq(embed_size=96, num_hiddens=256)

    output = model(data)

    print(output.shape)