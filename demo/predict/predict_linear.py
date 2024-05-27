"""
Predict script for Synthetic data

* @auther: xuan
* @email: 1920425406@qq.com
* @date: 2023-12-07
"""

import sys
sys.path.append('C:\LemonLover\WorkSpace\DL\TimeSeries-Predict')

import torch
import random
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from dataset.synthetic_dataset import SyntheticDataset
from model.linear import Model
import config


def lineArg():
    """
    define the plot format
    """
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan', 'black', 'indianred', 'brown', 'firebrick', 'maroon', 'darkred', 'red', 'sienna', 'chocolate', 'yellow', 'olivedrab', 'yellowgreen', 'darkolivegreen', 'forestgreen', 'limegreen', 'darkgreen', 'green', 'lime', 'seagreen', 'mediumseagreen', 'darkslategray', 'darkslategrey', 'teal', 'darkcyan', 'dodgerblue', 'navy', 'darkblue', 'mediumblue', 'blue', 'slateblue', 'darkslateblue', 'mediumslateblue', 'mediumpurple', 'rebeccapurple', 'blueviolet', 'indigo', 'darkorchid', 'darkviolet', 'mediumorchid', 'purple', 'darkmagenta', 'fuchsia', 'magenta', 'orchid', 'mediumvioletred', 'deeppink', 'hotpink']
    markers = [".", ",", "o", "v", "^", "<", ">", "1", "2", "3", "4", "8", "s", "p", "P", "*", "h", "H", "+", "x", "X", "D", "d", "|", "_", 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    linestyle = ['--', '-.', '-']
    line_arg = {}
    line_arg['color'] = random.choice(colors)
    # line_arg['marker'] = random.choice(markers)
    line_arg['linestyle'] = random.choice(linestyle)
    # line_arg['linewidth'] = random.randint(1, 4)
    return line_arg


def plot(real, pred):

    # set the random seed
    random.seed(0)

    plt.figure(figsize=(10, 6))
    x = range(len(real))
    plt.plot(x, real, **lineArg(), label='real')
    plt.plot(x, pred, **lineArg(), label='pred')
    plt.legend()
    plt.show()

def main(arg_dict):

    # test dataset
    synthetic_test = SyntheticDataset(train=False)
    test_loader = DataLoader(
        dataset=synthetic_test,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        pin_memory=True
    )
    # model 
    linear = Model(config.WINDOW_SIZE, output_dim=1)
    checkpoints = torch.load(arg_dict["checkpoints_path"], map_location=torch.device('cpu'))
    linear.load_state_dict(checkpoints["model"])

    # eval mode
    linear.eval()

    y_hat_lst = []
    y_lst = []
    with torch.no_grad():
        for x, y in test_loader:
            y_hat = linear(x)
            y_hat_lst.extend(list(y_hat.flatten().numpy()))
            y_lst.extend(list(y.flatten().numpy()))

    plot(y_lst, y_hat_lst)


if __name__ == '__main__':

    arg_dict = {
        "model": 'linear',   # linear
        "checkpoints_path": './logs/linear/model_499.tar',
        "show": True,
        "savefig": True,
        "save": True,    # save data to excel or not
        "root_dir": './results',   # if "save" is True, the results will be saved to this dir
        "normed": 'standard'   # norm or standard
    }

    main(arg_dict)
        