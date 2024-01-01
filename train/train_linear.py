"""
Train linear model for Synthetic data

* @auther: xuan
* @email: 1920425406@qq.com
* @date: 2023-12-07
"""

import sys
sys.path.append('D:\Python_WorkSpace\DL\TimeSeries')

from torch import optim
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import importlib

from dataset.synthetic_dataset import SyntheticDataset
import config
from utils import build_log_folder, save_checkpoints


def train(train_loader, model, opt, criterion, epoch, device):
    """
    train for linear model
    """

    loop = tqdm(train_loader, total=len(train_loader), leave=False)
    loop.set_description(f'epoch:{epoch}')

    loss_lst = []

    for x, y in train_loader:
        # move data to gpu
        x = x.to(device)
        y = y.to(device)

        y_hat = model(x)
        loss = criterion(y_hat, y)

        # updata
        opt.zero_grad()
        loss.backward()
        opt.step()

        loss_lst.append(loss.detach().cpu().numpy())

    return (
        sum(loss_lst) / len(loss_lst)
    )


def train_for_rnn(train_loader, model, opt, criterion, epoch, device):
    """
    train for rnn model
    """

    loop = tqdm(train_loader, total=len(train_loader), leave=False)
    loop.set_description(f'epoch:{epoch}')

    loss_lst = []

    # initial hidden state
    hs = None

    for x, y in train_loader:
        # move data to gpu
        x = x.to(device)
        y = y.to(device)

        # x shape: (batch_size, time_steps) --> (time_steps, batch_size, 特征维度)
        x = x.T.unsqueeze(2)   # (20, 32, 1)

        y_hat, hs = model(x, hs)
        loss = criterion(y_hat, y)

        # updata
        opt.zero_grad()
        loss.backward()
        opt.step()

        loss_lst.append(loss.detach().cpu().numpy())

    return (
        sum(loss_lst) / len(loss_lst)
    )

        

def main(arg_dict):
    # load the dataset
    synthetic_train = SyntheticDataset()
    # dataloader
    train_loader = DataLoader(
        dataset=synthetic_train,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        pin_memory=True
    )

    # Model
    imported_model = importlib.import_module('model.' + arg_dict["model"])
    model = imported_model.Model(input_dim=config.WINDOW_SIZE, output_dim=1).to(config.DEVICE)
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    # loss function
    criterion = nn.MSELoss()
    # 日志和参数保存
    log_dir = build_log_folder()
    writer = SummaryWriter(log_dir)

    # Training mode
    model.train()

    # Start training
    for epoch in range(config.NUM_EPOCHS):

        # train for epoch
        loss = train(train_loader, model, optimizer, criterion, epoch, device=config.DEVICE)

        # 保存模型参数
        if (epoch+1) % 100 == 0:
            checkpoints = {
                'model': model.state_dict(),
                'model_opt': optimizer.state_dict(),
            }
            save_checkpoints(checkpoints, log_dir, epoch)

        writer.add_scalar(tag='ModelLoss', scalar_value=loss, global_step=epoch)


if __name__ == '__main__':

    arg_dict = {
        "model": 'linear'    # linear
    }
    main(arg_dict)
