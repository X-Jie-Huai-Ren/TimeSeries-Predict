"""
the dataset process for forecasting

* @auther: xuan
* @email: 1920425406@qq.com
* @date: 2023-12-07
"""

import sys
sys.path.append('D:\Python_WorkSpace\DL\TimeSeries')

import numpy as np
import torch
from torch.utils.data import Dataset
from utils import trend, seasonality, noise


# the synthetic dataset
class SyntheticDataset(Dataset):
    """
    Through the synthetic data, for test
    """
    def __init__(self, window_size=20, train=True) -> None:

        self.window_size = window_size
        split_time = 1000

        # time, baseline, amplitude, slope, noise_level
        time = np.arange(4 * 365 + 1, dtype="float32")
        baseline = 10
        amplitude = 40
        slope = 0.05
        noise_level = 5

        # Generate the series data
        self.series = baseline + trend(time, slope) + seasonality(time, period=365, amplitude=amplitude)
        self.series += noise(time, noise_level, seed=42)

        # Split train_dataset and test_dataset
        if train:
            self.series = self.series[:split_time]
        else:
            self.series = self.series[split_time:]

    def __len__(self):
        """
        the length of dataset
        """
        return len(self.series) - self.window_size
    
    def __getitem__(self, index):

        # features
        features = self.series[index:index+self.window_size]
        # label
        label = self.series[index+self.window_size].reshape((1,))
        
        return torch.FloatTensor(features), torch.FloatTensor(label)
    

        
