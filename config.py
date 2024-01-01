"""
The Hyper Parameters Setup
"""

import torch

# Hyper Parameters
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 32
WINDOW_SIZE = 20
LEARNING_RATE = 2e-3
NUM_EPOCHS = 5000
LOAD_MDEOL_FILE = './logs'
FEATURE_DIM = 1