"""
testing the basic setup of a model script using a model with a single affine layer
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from training_utils import train
from layers import Flatten
from read_in_data import AmazonDataset


if __name__ == '__main__':
    ## cpu dtype
    dtype = torch.FloatTensor
    csv_path = '../../data/train.csv'
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
        Flatten(),
        nn.Linear(4*256*256, 17)
    )
    model.type(dtype)

    loss_fn = nn.BCELoss().type(dtype)
    optimizer = optim.Adam(model.parameters(), lr=5e-2)

    train(train_loader, model, loss_fn, optimizer, dtype,num_epochs=1, print_every=1e2)
