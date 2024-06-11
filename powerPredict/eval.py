"""
evaluate model which has been trained completely

* @author: xuan
* @email: 1920425406@qq.com 
* @date: 2024-06-04
"""

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing


from model.cnn import Model
from model.lstm import LSTMModel
from model.seq2seq_lstm import Seq2Seq
from model.Transformer import TransformerEncoder
from model.Transformer import TransformerDecoder
from model.Transformer import EncoderDecoder
import config

def prepare_eval_data(window_size):

    data = pd.read_excel('./data/power.xlsx').values

    # normaize the data like train data
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
    
    for item_data in splitted_data:
        feature = np.expand_dims(item_data[-window_size:], axis=0)
        features = np.concatenate([features, feature], axis=0)

    # real data
    real_data = pd.read_excel('./data/real.xlsx').values
    real_data = real_data[:, 2:]
    labels = np.zeros((1, 3, 96))
    for i in range(0, real_data.shape[0], 3):
        label = np.expand_dims(real_data[i:i+3], axis=0)
        labels = np.concatenate([labels, label], axis=0)
        
    # remove the first rows
    features, labels = features[1:], labels[1:]

    # transform data dtype: object --> float32
    features = np.array(features, dtype=np.float32)
    labels = np.array(labels, dtype=np.float32)

    return torch.FloatTensor(features), torch.FloatTensor(labels), mean, scale



def plot(y_preds, y, id):
    # tensor --> ndarray
    y_preds = y_preds.detach().numpy() # (3. 96)
    y = y.detach().numpy()  # (3, 96)

    colors = ['r', 'g', 'b']

    # plot
    x = np.arange(96)
    plt.figure(figsize=(10, 6))
    for i, (y_pred, y_real, color) in enumerate(zip(y_preds, y, colors)):
        plt.plot(x, y_pred, color=color, marker='o', label=f"5.{i+1}_pred")
        plt.plot(x, y_real, color=color, marker='>', label=f"5.{i+1}_real")
    plt.xlabel('power')
    plt.ylabel('time')
    plt.legend()
    plt.title(f"number_{id}")
    plt.savefig(f"./results/number_{id}.png")



def main(arg_dict):

    # eval data
    features, labels, mean, scale = prepare_eval_data(window_size=arg_dict["window_size"])

    mean = torch.FloatTensor(mean).unsqueeze(0)
    scale = torch.FloatTensor(scale).unsqueeze(0)

    # checkpoints
    checkpoints = torch.load(arg_dict["model_path"])

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

    model.load_state_dict(checkpoints)

    for i, (feature, y) in enumerate(zip(features, labels)):
        y = y.to(config.DEVICE)
        # 5.1
        feature = feature.to(config.DEVICE).unsqueeze(0)
        y_hat1 = model(feature)
        # scale the output to normalize
        y_pred1 = y_hat1 * scale + mean

        # 5.2
        feature = torch.concatenate([feature[0][1:], y_hat1], dim=0).unsqueeze(0)
        y_hat2 = model(feature)
        # scale the output to normalize
        y_pred2 = y_hat2 * scale + mean

        # 5.3
        feature = torch.concatenate([feature[0][1:], y_hat2], dim=0).unsqueeze(0)
        y_hat3 = model(feature)
        # scale the output to normalize
        y_pred3 = y_hat3 * scale + mean

        y_pred = torch.concatenate([y_pred1, y_pred2, y_pred3], dim=0)

        plot(y_pred, y, i)



if __name__ == '__main__':

    arg_dict = {
        "model": "cnn",
        "model_path": "./logs/[06-04]16.59.57/cnn/minloss.tar",
        "window_size": 7
    }

    main(arg_dict)

    