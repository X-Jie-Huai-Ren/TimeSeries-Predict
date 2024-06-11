"""
utility function

* @auther: xuan
* @email: 1920425406@qq.com
* @date: 2024-05-18
"""

import os
import torch
from torch import nn
from datetime import datetime, timedelta
import numpy as np
import config


# Standardize the data
def normalize(data):
    """
    :param data: shape(num_samples, output_dim)
    """
    # Calculate the mean and std
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    # Normalize
    data_normed = (data - mean) / (std + 1e-6)

    return data_normed, mean, std

def xavier_init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
    if type(m) == nn.RNN:
        for param in m._flat_weights_names:
            if "weight" in param:
                nn.init.xavier_uniform_(m._parameters[param])
    if type(m) == nn.LSTM:
        for param in m._flat_weights_names:
            if "weight" in param:
                nn.init.xavier_uniform_(m._parameters[param])



# 生成记录日志的文件夹
def build_log_folder(mode: str):
    cur_time = datetime.now() + timedelta(hours=0)  # hours参数是时区
    log_path_dir = os.path.join(config.LOAD_MDEOL_FILE, cur_time.strftime(f"[%m-%d]%H.%M.%S"))
    log_path_dir = os.path.join(log_path_dir, mode)
    # 若文件夹不存在，则创建
    if not os.path.exists(log_path_dir):
        os.makedirs(log_path_dir)
    return log_path_dir


# Save the checkpoints
def save_checkpoints(checkpoints, log_dir, epoch):
    """
    Params:
        checkpoints: 模型权重
        log_dir: 日志目录
        epoch: 当前训练的轮数
    """
    # 若文件夹不存在，则创建
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    path = log_dir + f'/model_{epoch}.tar'
    print('==> Saving checkpoints')
    torch.save(checkpoints, path)