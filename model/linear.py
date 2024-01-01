"""
Linear Model for Synthetic data

* @auther: xuan
* @email: 1920425406@qq.com
* @date: 2023-12-07
"""

from torch import nn

class Model(nn.Module):
    """
    Simple Linear Model
    """
    def __init__(self, input_dim, output_dim=1) -> None:
        """
        Params:
            input_dim: input dimension for input layer
        """
        super(Model,self).__init__()

        self.linear = nn.Sequential(
            nn.Linear(input_dim, output_dim)
        )

    def forward(self, x):

        output = self.linear(x)

        return output


