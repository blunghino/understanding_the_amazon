"""
testing the basic setup of a model script using a model with two conv layers
"""
import sys
import os.path

import torch
import torch.nn as nn
import torch.optim as optim
from scipy.io import savemat

from training_utils import train_epoch, validate_epoch
from layers import Flatten
from read_in_data import generate_train_val_dataloader, AmazonDataset
from pytorch_addons.pytorch_lr_scheduler.lr_scheduler import ReduceLROnPlateau


if __name__ == '__main__':
    try:
        from_pickle = int(sys.argv[1])
    except IndexError:
        from_pickle = 1
    ## cpu dtype
    dtype = torch.cuda.FloatTensor
    save_model_path = "conv_6-layer_state_dict.pkl"
    save_mat_path = "conv_6-layer_loss_and_acc.mat"
    csv_path = '../../data/train_v2.csv'
    img_path = '../../data/train-jpg'
    img_ext = '.jpg'

    dataset = AmazonDataset(csv_path, img_path, img_ext, dtype)

    train_loader, val_loader = generate_train_val_dataloader(
        dataset,
        batch_size=128,
        num_workers=0,
        use_fraction_of_data=0.2,
    )

    ## 6 conv layers
    model = nn.Sequential(
        ## 256x256
        nn.Conv2d(4, 16, kernel_size=3, stride=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(16),
        nn.Conv2d(16, 16, kernel_size=3, stride=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(16),
        nn.AdaptiveMaxPool2d(128),
        ## 128x128
        nn.Conv2d(16, 32, kernel_size=3, stride=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(32),
        nn.Conv2d(32, 32, kernel_size=3, stride=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(32),
        nn.AdaptiveMaxPool2d(64),
        ## 64x64
        nn.Conv2d(32, 64, kernel_size=3, stride=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(64),
        nn.Conv2d(64, 64, kernel_size=3, stride=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(64),
        nn.AdaptiveMaxPool2d(32),
        ## 32x32
        Flatten(),
        nn.Linear(64*32*32, 1024),
        nn.ReLU(inplace=True),
        nn.Linear(1024, 17)
    )
    model.type(dtype)

    ## set up optimization including hyperparams
    lr = 1e-3
    num_epochs = 5
    loss_fn = nn.MultiLabelSoftMarginLoss().type(dtype)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, patience=1, factor=0.5, min_lr=0.01*lr)

    acc_history = []
    loss_history = []
    ## don't load model params from file - instead retrain the model
    if not from_pickle:
        for epoch in range(num_epochs):
            print("Begin epoch {}/{}".format(epoch+1, num_epochs))
            epoch_loss = train_epoch(train_loader, model, loss_fn, optimizer,
                                       dtype, print_every=10)
            scheduler.step(epoch_loss[-1], epoch)
            ## f2 score for validation dataset
            acc = validate_epoch(model, val_loader, dtype)
            acc_history.append(acc)
            loss_history += epoch_loss
            print("END epoch {}/{}: F2 score = {:.02f}".format(epoch+1, num_epochs, acc))
        ## serialize model data and save as .pkl file
        torch.save(model.state_dict(), save_model_path)
        print("model saved as {}".format(os.path.abspath(save_model_path)))
        ## save loss and accuracy as .mat file
        savemat(save_mat_path, {"acc": acc_history, "loss": loss_history, "num_epochs": num_epochs})
    ## load model params from file
    else:
        state_dict = torch.load(save_model_path,
                                map_location=lambda storage, loc: storage)
        model.load_state_dict(state_dict)
        print("model loaded from {}".format(os.path.abspath(save_model_path)))
