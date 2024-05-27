"""
utility function

* @auther: xuan
* @email: 1920425406@qq.com
* @date: 2024-05-18
"""

import os
import torch
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
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


# 归一化数据
def normalize1(data):
    """
    :param data: shape(num_samples, output_dim)
    """
    # the maximum/minimum
    maximum = np.max(data)
    minimum = np.min(data)
    # Normalize
    data_normed = (data - minimum) / (maximum - minimum)

    return data_normed, maximum, minimum


# 生成记录日志的文件夹
def build_log_folder():
    cur_time = datetime.now() + timedelta(hours=0)  # hours参数是时区
    log_path_dir = os.path.join(config.LOAD_MDEOL_FILE, cur_time.strftime(f"[%m-%d]%H.%M.%S"))
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