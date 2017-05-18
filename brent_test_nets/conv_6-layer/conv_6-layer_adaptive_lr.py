"""
testing the basic setup of a model script using a model with two conv layers
"""
import sys
import os.path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch import np

from training_utils import train_epoch, validate_epoch, test_model
from layers import Flatten
from read_in_data import generate_train_val_dataloader, AmazonDataset, AmazonTestDataset
from pytorch_addons.pytorch_lr_scheduler.lr_scheduler import ReduceLROnPlateau
from plotting_tools import save_accuracy_and_loss_mat


if __name__ == '__main__':
    ## command line arg to determine retraining vs loading model
    try:
        from_pickle = int(sys.argv[1])
    except IndexError:
        from_pickle = True
    ## command line arg to set GPU/CPU
    try:
        use_cuda = int(sys.argv[2])
    except IndexError:
        use_cuda = True
    ############################### SETTINGS ###################################
    ## only need to change things in this part of the code
    root = "conv_6-layer" # name of model
    save_model_path = "{}_state_dict.pkl".format(root)
    save_mat_path = "{}_loss_and_acc.mat".format(root)
    csv_path = '../../data/train_v2.csv'
    img_path = '../../data/train-jpg'
    img_ext = '.jpg'
    ## dataloader params
    batch_size = 128
    use_fraction_of_data = 0.2 # 1 to train on full data set
    ## optimization hyperparams
    lr = 1e-3
    num_epochs = 7
    adaptive_lr_patience = 0 # scale lr after loss plateaus for "patience" epochs
    adaptive_lr_factor = 0.1 # scale lr by this factor
    ## whether to generate predictions on test
    run_test = True
    test_csv_path = "../../data/sample_submission.csv"
    test_img_path = "../../data/test-jpg"
    test_results_csv_path = "{}_results.csv".format(root)
    ############################################################################
    ## cpu/gpu settings
    if use_cuda:
        dtype = torch.cuda.FloatTensor
        num_workers = 0
    else:
        dtype = torch.FloatTensor
        num_workers = 4

    dataset = AmazonDataset(csv_path, img_path, img_ext, dtype)

    train_loader, val_loader = generate_train_val_dataloader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        use_fraction_of_data=use_fraction_of_data,
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

    ## set up optimization
    loss_fn = nn.MultiLabelSoftMarginLoss().type(dtype)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, patience=adaptive_lr_patience,
                                  cooldown=2, verbose=1, min_lr=1e-5*lr,
                                  factor=adaptive_lr_factor)

    acc_history = []
    loss_history = []
    ## don't load model params from file - instead retrain the model
    if not from_pickle:
        for epoch in range(num_epochs):
            print("Begin epoch {}/{}".format(epoch+1, num_epochs))
            epoch_losses = train_epoch(train_loader, model, loss_fn, optimizer,
                                       dtype, print_every=10)
            scheduler.step(np.mean(epoch_losses), epoch)
            ## f2 score for validation dataset
            acc = validate_epoch(model, val_loader, dtype)
            ## store results
            acc_history.append(acc)
            loss_history += epoch_losses
            print("END epoch {}/{}: F2 score = {:.02f}".format(epoch+1, num_epochs, acc))
        ## serialize model data and save as .pkl file
        torch.save(model.state_dict(), save_model_path)
        print("model saved as {}".format(os.path.abspath(save_model_path)))
        ## save loss and accuracy as .mat file
        save_accuracy_and_loss_mat(save_mat_path, acc_history, loss_history, num_epochs)
    ## load model params from file
    else:
        state_dict = torch.load(save_model_path,
                                map_location=lambda storage, loc: storage)
        model.load_state_dict(state_dict)
        print("model loaded from {}".format(os.path.abspath(save_model_path)))
    ## generate predictions on test data set
    if run_test:
        test_dataset = AmazonTestDataset(test_csv_path, test_img_path, img_ext, dtype)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)
        test_preds = test_model(model, test_loader, train_loader.dataset.mlb, dtype,
                                out_file_name=test_results_csv_path)

