"""

"""

from typing import Any
import pandas as pd
import numpy as np


import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from utils import normalize


np.random.seed(0)


class LoadDataset(Dataset):
    """"""
    def __init__(self, train=None) -> None:
        # load data
        self.data = pd.read_csv('./train_data.csv').values

        # features
        self.features = self.data[:, :-1]
        # labels
        self.labels = self.data[:, -1]

        # normalize
        self.features, self.mean, self.std = normalize(self.features)

        # the indices of test data
        idx = np.random.choice(np.arange(0, len(self.features)), size=int(0.2*len(self.features)), replace=False)
        # create the mask
        mask = np.ones(self.features.shape[0], dtype=bool)
        mask[idx] = False

        if train:
            self.features = self.features[mask]
            self.labels = self.labels[mask]
        else:
            self.features = self.features[idx]
            self.labels = self.labels[idx]

    def __len__(self):

        return len(self.features)
    

    def __getitem__(self, index) -> Any:
        
        return self.features[index], self.labels[index]
    



class LoadDatasetTS(Dataset):
    """
    the load dataset for time series
    """
    def __init__(self, window_size=24, train=None, label_in_feature=False, prob=0.2) -> None:

        self.window_size = window_size

        # load data
        self.data = pd.read_csv('./data/train_data.csv').values

        # features
        self.features = self.data[:, 10:] if label_in_feature else self.data[:, 10:-1]
        # labels
        self.labels = self.data[:, -1]

        # normalize
        self.features, self.mean, self.std = normalize(self.features)

        if train:
            self.features = self.features[:int((1 - prob) * len(self.features)), :]
            self.labels = self.labels[:int((1 - prob) * len(self.labels))]
        else:
            self.features = self.features[int((1 - prob) * len(self.features)):, :]
            self.labels = self.labels[int((1 - prob) * len(self.labels)):]

        # reshape labels: (num_samples, ) --> (window_size:, 1)
        self.labels = self.labels[window_size:]
        self.labels = self.labels.reshape((-1, 1))


    def __len__(self):

        return len(self.labels)
    

    def __getitem__(self, index) -> Any:
        """
        Returns:
            -features: shape(window_size, embedd_size)
        """
        
        features = self.features[index:index+self.window_size, :]
        labels = self.labels[index]

        return torch.FloatTensor(features), torch.FloatTensor(labels)
            

        
    




if __name__ == '__main__':

    dataset_train = LoadDatasetTS(train=True)

    dataloader = DataLoader(
        dataset=dataset_train,
        batch_size=4,
        shuffle=False,
        drop_last=True,
        pin_memory=True
    )

    for x, y in dataloader:
        x = x.to('cpu')
        print(y.shape)
        break