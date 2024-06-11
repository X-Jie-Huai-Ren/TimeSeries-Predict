"""
train script, for time-series cnn model

* @author: xuan
* @email: 1920425406@qq.com
* @date: 2024-05-21
"""

import os
import torch
import argparse
from tqdm import tqdm
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


from utils import build_log_folder, xavier_init_weights
from dataset import load_data
from model.cnn import Model
from model.lstm import LSTMModel
from model.seq2seq_lstm import Seq2Seq
from model.Transformer import TransformerEncoder
from model.Transformer import TransformerDecoder
from model.Transformer import EncoderDecoder
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
        self.loss_func = nn.MSELoss(reduction='sum')

    
    def __train_for_epoch(self, epoch):
        loop = tqdm(self.train_loader, total=len(self.train_loader), leave=False)
        loop.set_description(f'train epoch: {epoch}')

        loss_lst = []
        r2_lst = []

        for features, labels in loop:

            # data to device
            features = features.to(config.DEVICE)
            labels = labels.to(config.DEVICE)
            
            # get the prediction y_hat
            y_hat = self.model(features)
            # calculate the loss
            loss = self.loss_func(y_hat, labels)
            # calculate the r2
            r2 = self.cal_r2(y_hat, labels)
            # optimize model
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            loss_lst.append(loss)
            r2_lst.append(r2)

        return sum(loss_lst) / len(loss_lst), sum(r2_lst) / len(r2_lst)
    

    def __eval_for_interval(self, epoch):
        loop = tqdm(self.test_loader, total=len(self.test_loader), leave=False)
        loop.set_description(f'test epoch: {epoch}')

        loss_lst = []
        r2_lst = []

        for features, labels in loop:

            # data to device
            features = features.to(config.DEVICE)
            labels = labels.to(config.DEVICE)
            
            # get the prediction y_hat
            y_hat = self.model(features)
            # calculate the loss
            loss = self.loss_func(y_hat, labels)
            # calculate the r2
            r2 = self.cal_r2(y_hat, labels)

            loss_lst.append(loss)
            r2_lst.append(r2)

        return sum(loss_lst) / len(loss_lst), sum(r2_lst) / len(r2_lst)
    

    def cal_r2(self, y_hat, y):
        """
        y_hat: shape(batch_size, 96)
        y: shape(batch_size, 96)
        """
        y_avg = torch.mean(y, dim=1)
        y1 = torch.sum(torch.pow(y - y_hat, 2), dim=1)
        y2 = torch.sum(torch.pow(y - y_avg.unsqueeze(1).repeat(1, 96), 2), dim=1)

        r2 = 1 - torch.mean(y1 / y2)

        return r2


    def train(self):
        """"""
        i = 0
        loss_max = 1500

        # train for epoch
        for epoch in range(self.arg_dict["num_epochs"]):
            train_loss, train_r2 = self.__train_for_epoch(epoch)

            if (epoch+1) % 10 == 0:
                eval_loss, eval_r2 = self.__eval_for_interval(epoch)
                self.writer.add_scalar(tag='Eval/loss', scalar_value=eval_loss, global_step=i)
                self.writer.add_scalar(tag='Eval/r2', scalar_value=eval_r2, global_step=i)
                i += 1

            # save the model
            if train_loss < loss_max:
                loss_max = train_loss
                self._save_model(epoch, min=True)
            if epoch > 500 and (epoch+1) % 100 == 0:
                self._save_model(epoch)

            self.writer.add_scalar(tag='Train/loss', scalar_value=train_loss, global_step=epoch)
            self.writer.add_scalar(tag='Train/r2', scalar_value=train_r2, global_step=epoch)

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
    train_dataset, test_dataset, mean, std = load_data('./data/power.xlsx', window_size=arg_dict["window_size"])
    # dataloader
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=arg_dict['batch_size'],
        shuffle=True,
        pin_memory=True
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=arg_dict['batch_size'],
    )

    # model and optimizer
    if arg_dict["model"] == "cnn":
        model = Model(window_size=arg_dict["window_size"], in_features=96).to(config.DEVICE)
    elif arg_dict["model"] == "lstm":
        model = LSTMModel(input_size=96, hidden_size=128, num_layers=1)
    elif arg_dict["model"] == "seq2seq":
        model = Seq2Seq(input_size=96, output_size=96, hidden_size=512, num_layers=1, pred_len=1, window_size=arg_dict["window_size"])
    elif arg_dict["model"] == "transformer":
        encoder = TransformerEncoder(vocab_size=96).to(config.DEVICE)
        decoder = TransformerDecoder(vocab_size=96).to(config.DEVICE)
        model = EncoderDecoder(encoder, decoder).to(config.DEVICE)
    # Initialize the model weights
    model.apply(xavier_init_weights)
    opt = optim.Adam(model.parameters(), lr=arg_dict['lr'])

    # 日志和参数保存
    log_dir = build_log_folder(arg_dict["model"])
    arg_dict['log_dir'] = log_dir

    # Training mode
    model.train()

    # trainer
    trainer = Trainer(arg_dict, train_loader, test_loader, model, opt)

    trainer.train()


if __name__ == '__main__':

    arg_dict = {
        "window_size": 7,
        "batch_size": 32,
        "lr": 1e-3,
        "num_epochs": 1000
    }

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="cnn")
    args = parser.parse_args()

    assert args.model in ['cnn', 'lstm', 'seq2seq', 'transformer'], "The args 'model' must be one of ['cnn', 'lstm', 'seq2seq', 'transformer']"

    arg_dict["model"] = args.model

    print(f"the {args.model} model is training...")

    main(arg_dict)