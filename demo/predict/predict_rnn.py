"""
Predict script for Synthetic data, for RNN model

* @auther: xuan
* @email: 1920425406@qq.com
* @date: 2024-01-01
"""

import sys
sys.path.append('D:\Python_WorkSpace\DL\TimeSeries')


import torch
import random
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from dataset.synthetic_dataset import SyntheticDataset
from model.linear import Model
import config