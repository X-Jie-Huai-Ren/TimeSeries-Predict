"""
load the data as dataset

* @author: xuan
* @email: 1920425406@qq.com 
* @date: 2024-06-01
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Any
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

from utils import normalize

class PowerDataset(Dataset):
    """
    load data to dataset
    """
    def __init__(self, features: np.ndarray, labels: np.ndarray) -> None:
        super().__init__()
    
        self.features = features
        self.labels = labels

        # transform data dtype: object --> float32
        self.features = np.array(self.features, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.float32)

        # don't need normalize data

    def __len__(self):
        
        return self.features.shape[0]
    
    def __getitem__(self, index) -> Any:
        
        return self.features[index], self.labels[index]



def reshape_power_data(data: np.ndarray, window_size: int):
    """
    reshape demonsion of data

    Params:
        data: 
            the power data of nine stations
            the data for each station includes data from 2018.01.01 to 2019.04.30
            the time resolution is 15 min
        - window_size:
            using previous 'window_size' days' data to predict next day
    """
    # normaize the data
    scaler = preprocessing.StandardScaler().fit(data[:, 2:])
    mean = scaler.mean_
    scale = scaler.scale_
    data[:, 2:] = scaler.transform(data[:, 2:])
    # split the data according to station
    stations = set(data[:, 0])
    splitted_data = [data[data[:, 0] == station] for station in stations]
    # remove the useless cols
    splitted_data = [item[:, 2:] for item in splitted_data]

    features = np.zeros((1, window_size, 96))
    labels = np.zeros((1, 96))

    # split data into features and labels
    for item_data in tqdm(splitted_data):
        # the number of samples for this station
        # previous 6 days --> next day
        num_samples = item_data.shape[0] - window_size
        for i in range(num_samples):
            feature = np.expand_dims(item_data[i: i+window_size], axis=0)
            label = np.expand_dims(item_data[i+window_size], axis=0)
            # add
            features = np.concatenate([features, feature], axis=0)
            labels = np.concatenate([labels, label], axis=0)

    # remove the first rows
    features, labels = features[1:], labels[1:]
    
    return features, labels, mean, scale



def load_data(data_path1: str, window_size: int = 7, weather_in_features: bool = False, data_path2 = None):
    """
    load the power data, and make a Dataset

    Params:
        - data_path1: 
            the file path of power data
        - window_size:
            using previous 'window_size' days' data to predict next day
        - weather_in_features: 
            if True, the weather data is input features, if False, is not; 
            default is False
            if True, you must give the parameters 'data_path2'
        - data_path2:
            the file path of weather data
            default is None
    """
    print("The dataset is loading...")
    # power data
    power_data = pd.read_excel(data_path1).values
    # reshape the dimension of power data
    features, labels, mean, std = reshape_power_data(power_data, window_size)
    # split train and test
    train_x, test_x, train_y, test_y = train_test_split(features, labels, train_size=0.8, test_size=0.2, random_state=2, shuffle=True)

    # load the dataset 
    train_dataset = PowerDataset(train_x, train_y)
    test_dataset = PowerDataset(test_x, test_y)

    # train_loader = DataLoader(
    #     train_dataset,
    #     batch_size=32
    # )
    # for x, y in train_loader:
    #     print(x.shape)
    #     break
    
    return train_dataset, test_dataset, mean, std

# load_data('./data/power.xlsx')
