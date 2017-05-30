"""
testing the basic setup of a model script using a model with two conv layers
"""
import sys
import os.path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from training_utils import train, validate_epoch
from layers import Flatten
from read_in_data import AmazonDataset
from resnet_originals import *
from itertools import chain


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
    img_ext = '.jpg'
    training_dataset = AmazonDataset(csv_path, img_path, img_ext, dtype)
    ## loader
    train_loader = DataLoader(
        training_dataset,
        batch_size=10,
        shuffle=True,
        num_workers=1, # 1 for CUDA
    )

    # val_loader = DataLoader()

    ## simple linear model
    model = resnet18(pretrained=True)
    model.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
    model.fc = nn.Linear(model.inplanes, 17)
    model.type(dtype)

    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True
    for param in model.conv1.parameters():
        param.requires_grad = True

    loss_fn = nn.MultiLabelSoftMarginLoss().type(dtype)
    optimizer = optim.Adam(chain(model.fc.parameters(), model.conv1.parameters()), lr=1e-3)
    ## don't load model params from file - instead retrain the model
    if not from_pickle:
        train(train_loader, model, loss_fn, optimizer, dtype, print_every=1)
        ## serialize model data and save as .pkl file
        torch.save(model.state_dict(), save_model_path)
        print("model saved as {}".format(os.path.abspath(save_model_path)))
    ## load model params from file
    else:
        state_dict = torch.load(save_model_path,map_location=lambda storage, loc: storage)
        model.load_state_dict(state_dict)
        print("model loaded from {}".format(os.path.abspath(save_model_path)))

    train_acc_loader = DataLoader(training_dataset, batch_size=200, shuffle=True, num_workers=6)
    acc = validate_epoch(model, train_acc_loader, dtype)
    print(acc)
