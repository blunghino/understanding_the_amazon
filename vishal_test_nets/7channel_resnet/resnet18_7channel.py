"""
testing the basic setup of a model script using a model with two conv layers
"""
import sys
import os.path

from torchvision.models import resnet18
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch import np
from matplotlib import pyplot as plt

from pytorch_addons.pytorch_lr_scheduler.lr_scheduler import ReduceLROnPlateau

from training_utils import train_epoch, validate_epoch, test_model
from read_in_data import (generate_train_val_dataloader, ResnetTrainDataset,
                          ResnetTestDataset)
from plotting_tools import save_accuracy_and_loss_mat, image_and_labels
from optimize_cutoffs import get_scores, optimize_F2


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

    root = "resnet18_7channel" # name of model
    save_model_path = "{}_state_dict.pkl".format(root)
    save_mat_path = "{}_training_data.mat".format(root)
    csv_path = '../../data/train_v2.csv'
    img_path = '../../data/train_combo'
    img_ext = '.npy'
    save_every = 5
    ## dataloader params
    batch_size = 128
    use_fraction_of_data = 1 # 1 to train on full data set
    ## optimization hyperparams
    lr = 1e-3
    num_epochs = 20
    reg = 1e-3
    adaptive_lr_patience = 0 # scale lr after loss plateaus for "patience" epochs
    adaptive_lr_factor = 0.1 # scale lr by this factor
    ## whether to generate predictions on test
    run_test = True
    test_csv_path = "../../data/sample_submission_v2.csv"
    test_img_path = "../../data/test_combo"
    test_results_csv_path = "{}_results.csv".format(root)
    ############################################################################

    ## cpu/gpu setup
    if use_cuda:
        dtype = torch.cuda.FloatTensor
        num_workers = 0
    else:
        dtype = torch.FloatTensor
        num_workers = 4

    NPY_7CHANNEL_MEAN = [0.68856319,  0.65945722,  0.70117001, 0.07612355,
                         0.06516739, 0.04691964, 0.09764017]
    NPY_7CHANNEL_STD = [0.05338322,  0.04247037,  0.03543733, 0.00608935,
                        0.00623353, 0.00691525, 0.01309933]

    dataset = ResnetTrainDataset(csv_path, img_path, dtype, img_ext=img_ext,
                                 channel_means=NPY_7CHANNEL_MEAN,
                                 channel_stds=NPY_7CHANNEL_STD)

    train_loader, val_loader = generate_train_val_dataloader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        use_fraction_of_data=use_fraction_of_data,
    )

    ## pretrained resnet
    model = resnet18(pretrained=False)
    ## resize last fully connected layer to match our problem
    model.fc = nn.Linear(model.fc.in_features, 17)
    ## resize the first conv layer to match our problem
    model.conv1 = nn.Conv2d(7, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.type(dtype)
    model=torch.load("resnet18_7channel_state_dict_epoch-35.pkl")
    
    sigmoid_threshold = 0.25
    loss_fn = nn.MultiLabelSoftMarginLoss().type(dtype)

    ## don't load model params from file - instead retrain the model
    if not from_pickle:

        train_acc_history = []
        val_acc_history = []
        loss_history = []

        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=reg)
        scheduler = ReduceLROnPlateau(optimizer, patience=adaptive_lr_patience,
                                        cooldown=2, verbose=1, min_lr=1e-5*lr,
                                          factor=adaptive_lr_factor)

        for epoch in range(1, num_epochs+1):
            print("Begin epoch {}/{}".format(epoch, num_epochs))
            epoch_losses, epoch_f2 = train_epoch(train_loader, model, loss_fn,
                                                 optimizer, dtype,
                                                 sigmoid_threshold=sigmoid_threshold,
                                                 print_every=10)
            scheduler.step(np.mean(epoch_losses), epoch)
            ## f2 score for validation dataset
            f2_acc = validate_epoch(model, val_loader, dtype,
                                    sigmoid_threshold=sigmoid_threshold)
            ## store results
            train_acc_history += epoch_f2
            val_acc_history.append(f2_acc)
            loss_history += epoch_losses
            ## overwrite the model .pkl file every epoch
            torch.save(model.state_dict(), save_model_path)
            save_accuracy_and_loss_mat(save_mat_path, train_acc_history,
                                       val_acc_history, loss_history, epoch,
                                       lr=optimizer.param_groups[-1]['lr'])
            ## checkpoints
            if save_every and not epoch % save_every:
                save_epoch_path = "{}_epoch-{:d}.pkl".format(
                                           save_model_path.split('.')[0], epoch)
                torch.save(model.state_dict(), save_epoch_path)
                print("checkpoint saved as {}".format(os.path.abspath(save_epoch_path)))

            print("END epoch {}/{}: validation F2 score = {:.02f}".format(
                  epoch, num_epochs, f2_acc))
        ## serialize model data and save as .pkl file
        print("model saved as {}".format(os.path.abspath(save_model_path)))


    ## load model params from file
    else:
        state_dict = torch.load(save_model_path,
                                map_location=lambda storage, loc: storage)
        model.load_state_dict(state_dict)
        print("model loaded from {}".format(os.path.abspath(save_model_path)))

    ## generate predictions on test data set
    if run_test:
        ## first optize sigmoid thresholds
        print("optimizing sigmoid cutoffs for each class")
        sig_scores, y_array = get_scores(model, train_loader, dtype)
        sigmoid_threshold = optimize_F2(sig_scores, y_array)
        print("optimal thresholds: ", sigmoid_threshold)

        print("running model on test set")
        test_dataset = ResnetTestDataset(test_csv_path, test_img_path, dtype,
                                         img_ext=img_ext,
                                         channel_means=NPY_7CHANNEL_MEAN,
                                         channel_stds=NPY_7CHANNEL_STD)
        test_loader = DataLoader(test_dataset, batch_size=batch_size,
                                 num_workers=num_workers)
        test_preds = test_model(model, test_loader, train_loader.dataset.mlb,
                                dtype, sigmoid_threshold=sigmoid_threshold,
                                out_file_name=test_results_csv_path)
        print("test set results saved as {}".format(os.path.abspath(test_results_csv_path)))
