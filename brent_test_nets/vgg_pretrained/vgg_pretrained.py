"""
testing the basic setup of a model script using a model with two conv layers
"""
import sys
import os.path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models
from torchvision import transforms

from training_utils import validate_epoch
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
    img_ext = '.jpg'

    transform_list = [transforms.Scale(224)]
    training_dataset = AmazonDataset(csv_path, img_path, img_ext, dtype,
                                     transform_list=transform_list, three_band=True)
    ## loader
    train_loader = DataLoader(
        training_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=4, # 1 for CUDA
    )

    model = torchvision.models.vgg11(pretrained=True)
    model.type(dtype)

    loss_fn = nn.MultiLabelSoftMarginLoss().type(dtype)

    acc, loss = validate_epoch(model, train_loader, loss_fn, dtype)

    print(acc, loss)
