"""
hyper parameters

* @auther: xuan
* @email: 1920425406@qq.com
* @date: 2024-05-18
"""

import torch


LOAD_MDEOL_FILE = './logs'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'