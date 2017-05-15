"""
testing the basic setup of a model script using a model with two conv layers
"""
import os.path 
import sys

import torch
import torch.nn as nn
import torch.optim as optim 
from torch.utils.data import DataLoader

from training_utils import train
from layers import Flatten
from read_in_data import AmazonDataset


if __name__ == '__main__':
    try:
        from_pickle = int(sys.argv[1])
    except IndexError:
        from_pickle = 1
    ## cpu dtype
    dtype = torch.FloatTensor
    save_model_path = "model_state_dict.pkl"
    csv_path = '../../data/train_v2.csv'
    img_path = '../../data/train-jpg'
    training_dataset = AmazonDataset(csv_path, img_path, dtype)
    ## loader
    train_loader = DataLoader(
        training_dataset,
        batch_size=256,
        shuffle=True,
        num_workers=4 # 1 for CUDA
        # pin_memory=True # CUDA only
    )
    ## simple linear model
    model = nn.Sequential(
        nn.Conv2d(4, 16, kernel_size=3, stride=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(16),
        nn.AdaptiveMaxPool2d(128),
        nn.Conv2d(16, 32, kernel_size=3, stride=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(32),
        nn.AdaptiveMaxPool2d(64),
        Flatten(),
        nn.Linear(32*64*64, 1024),
        nn.ReLU(inplace=True),
        nn.Linear(1024, 17)
    )
    model.type(dtype)

    loss_fn = nn.BCELoss().type(dtype)
    optimizer = optim.Adam(model.parameters(), lr=5e-2)
    ## don't load model params from file - instead retrain the model
    if not from_pickle:
        train(train_loader, model, loss_fn, optimizer, dtype)
        ## serialize model data and save as .pkl file
        torch.save(model.state_dict(), save_model_path)
        print("model saved as {}".format(os.path.abspath))
    ## load model params from file
    else:
        state_dict = torch.load(save_model_path)
        model.load_state_dict(state_dict)
        
