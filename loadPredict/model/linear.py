"""
multi linear regression

* @auther: xuan
* @email: 1920425406@qq.com
* @date: 2024-05-20
"""


from torch import nn



class Model(nn.Module):
    """
    linear model for load prediction
    """
    def __init__(self, input_dim: int, output_dim: int) -> None:
        """
        Initialize function

        Params:
            - input_dim: the number of input units
            - output_dim: the number of output units
        """
        super(Model, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        # linear model
        self.linear = nn.Linear(self.input_dim, self.output_dim)


    def forward(self, x):

        return self.linear(x)