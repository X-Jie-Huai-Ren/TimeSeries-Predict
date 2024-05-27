"""
train script, no time-series

* @author: xuan
* @email: 1920425406@qq.com
* @date: 2024-05-20
"""

import importlib
from torch.utils.data import DataLoader
from torch import optim

from dataset import LoadDataset
from utils import build_log_folder
import config


class Trainer:
    """
    train manager
    """
    def __init__(self, arg_dict, train_loader, test_loader, model, opt) -> None:
        pass



def main(arg_dict):
    
    # load the dataset
    dataset_train = LoadDataset(train=True)
    dataset_test = LoadDataset(train=False)
    # dataloader
    train_loader = DataLoader(
        dataset=dataset_train,
        batch_size=arg_dict['batch_size'],
        shuffle=True,
        pin_memory=True
    )
    test_loader = DataLoader(
        dataset=dataset_test,
        batch_size=arg_dict['batch_size'],
        shuffle=False
    )

    # model and optimizer
    imported_model = importlib.import_module("model." + arg_dict['model'])
    model = imported_model.Model(input_dim=12, output_dim=1).to(config.DEVICE)
    opt = optim.Adam(model.parameters(), lr=arg_dict['lr']).to(config.DEVICE)

    # 日志和参数保存
    log_dir = build_log_folder()
    arg_dict['log_dir'] = log_dir

    # Training mode
    model.train()









if __name__ == '__main__':

    arg_dict = {
        'model': 'linear',   # 'linear' or 'mlp' or ...
        'batch_size': 32
    }
