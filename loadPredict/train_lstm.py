"""
train script, for time-series lstm model

* @author: xuan
* @email: 1920425406@qq.com
* @date: 2024-05-21
"""

import os
import torch
from tqdm import tqdm
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


from utils import build_log_folder
from dataset import LoadDatasetTS
from model.lstm import LSTMModel
import config



class Trainer:
    """
    train manager
    """
    def __init__(self, arg_dict, train_loader, test_loader, model, opt) -> None:
        """"""
        self.arg_dict = arg_dict
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model = model
        self.opt = opt

        # tensorboard writer
        self.writer = SummaryWriter(self.arg_dict['log_dir'])

        # loss function
        self.loss_func = nn.MSELoss()

    
    def __train_for_epoch(self, epoch):
        loop = tqdm(self.train_loader, total=len(self.train_loader), leave=False)
        loop.set_description(f'train epoch: {epoch}')

        loss_lst = []

        for features, labels in loop:

            # data to device
            features = features.to(config.DEVICE)
            labels = labels.to(config.DEVICE)

            # change the dimensions of features for statisfying the LSTM
            features = features.permute(1, 0, 2)
            
            # get the prediction y_hat
            y_hat = self.model(features)
            # calculate the loss
            loss = self.loss_func(y_hat, labels)
            # optimize model
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            loss_lst.append(loss)

        return sum(loss_lst) / len(loss_lst)  
    

    def __eval_for_interval(self, epoch):
        loop = tqdm(self.test_loader, total=len(self.test_loader), leave=False)
        loop.set_description(f'test epoch: {epoch}')

        loss_lst = []

        for features, labels in loop:

            # data to device
            features = features.to(config.DEVICE)
            labels = labels.to(config.DEVICE)
            # change the dimensions of features for statisfying the Conv1d
            features = features.permute(1, 0, 2)
            
            # get the prediction y_hat
            y_hat = self.model(features)
            # calculate the loss
            loss = self.loss_func(y_hat, labels)

            loss_lst.append(loss)

        return sum(loss_lst) / len(loss_lst)



    def train(self):
        """"""
        i = 0
        loss_max = 1500

        # train for epoch
        for epoch in range(self.arg_dict["num_epochs"]):
            train_loss = self.__train_for_epoch(epoch)

            if (epoch+1) % 100 == 0:
                eval_loss = self.__eval_for_interval(epoch)
                self.writer.add_scalar(tag='eval_loss', scalar_value=eval_loss, global_step=i)
                i += 1

            # save the model
            if train_loss < loss_max:
                loss_max = train_loss
                self._save_model(epoch, min=True)
            if epoch > 200 and (epoch+1) % 100 == 0:
                self._save_model(epoch)

            self.writer.add_scalar(tag='Train loss', scalar_value=train_loss, global_step=epoch)

    def _save_model(self, epoch, min=False):
        """
        """
        if not os.path.exists(self.arg_dict['log_dir']):
            os.makedirs(self.arg_dict['log_dir'])
        checkpoints = self.model.state_dict()
        if min:
            path = self.arg_dict['log_dir'] + '/minloss.tar'
        else:
            path = self.arg_dict['log_dir'] + f'/model_{epoch}.tar'
        print(f'==> Saving checkpoints: {epoch}')
        torch.save(checkpoints, path)
            

def main(arg_dict):

    # load the dataset
    dataset_train = LoadDatasetTS(train=True, label_in_feature=True)
    dataset_test = LoadDatasetTS(train=False, label_in_feature=True)
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
    )

    # model and optimizer
    model = LSTMModel(input_size=3, hidden_size=64, num_layers=1).to(config.DEVICE)
    opt = optim.Adam(model.parameters(), lr=arg_dict['lr'])

    # 日志和参数保存
    log_dir = build_log_folder()
    arg_dict['log_dir'] = log_dir

    # Training mode
    model.train()

    # trainer
    trainer = Trainer(arg_dict, train_loader, test_loader, model, opt)

    trainer.train()


if __name__ == '__main__':

    arg_dict = {
        "model": "cnn",  
        "batch_size": 32,
        "lr": 1e-3,
        "num_epochs": 2000
    }

    main(arg_dict)