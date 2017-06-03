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

from pytorch_addons.pytorch_lr_scheduler.lr_scheduler import ReduceLROnPlateau
from training_utils import (train_epoch, validate_epoch, test_triple_resnet,
                            get_triple_resnet_val_scores)
from read_in_data import (triple_train_val_dataloaders, ResnetTrainDataset,
                          ResnetTestDataset)
from optimize_cutoffs import optimize_F2
from plotting_tools import save_accuracy_and_loss_mat


if __name__ == '__main__':
    ## command line arg to determine retraining vs loading model
    try:
        pickle_arg = int(sys.argv[1])
        ## set to all 1's or all 0's
        from_pickle = [pickle_arg, pickle_arg, pickle_arg]
    except IndexError:
        from_pickle = [1, 1, 1]
    ## command line arg to set GPU/CPU
    try:
        use_cuda = int(sys.argv[2])
    except IndexError:
        use_cuda = True

    ############################### SETTINGS ###################################
    ## only need to change things in this part of the script

    ## use this for customizing which models to retrain
    # from_pickle = [1, 0, 0]
    ## paths
    root = "triple_resnet18" # name of model
    csv_path = '../../data/train_v2.csv'
    img_paths = [
        '../../data/train-jpg',
        '../../data/train_inf',
        '../../data/train_grad'
    ]
    model_names = ['rgb', 'inf', 'grad']
    img_ext = '.jpg'
    ## dataloader params
    batch_size = 256
    use_fraction_of_data = 1 # 1 to train on full data set
    ## optimization hyperparams
    sigmoid_threshold = 0.25
    lr_1 = 1e-3
    num_epochs_1 = 3
    reg_1 = 0
    lr_2 = 1e-5
    num_epochs_2 = 12
    reg_2 = 5e-4
    adaptive_lr_patience = 0 # scale lr after loss plateaus for "patience" epochs
    adaptive_lr_factor = 0.1 # scale lr by this factor
    ## whether to generate predictions on test
    run_test = True
    test_csv_path = "../../data/sample_submission_v2.csv"
    test_img_paths = [
        "../../data/test-jpg",
        "../../data/test_inf",
        "../../data/test_grad",
    ]
    test_model_weights = (6., 1., 1.)
    test_results_csv_path = "{}_results.csv".format(root)
    ############################################################################

    ## cpu/gpu setup
    if use_cuda:
        dtype = torch.cuda.FloatTensor
        num_workers = 0
    else:
        dtype = torch.FloatTensor
        num_workers = 4

    datasets = [ResnetTrainDataset(csv_path, ip, dtype) for ip in img_paths]

    train_loaders, val_loaders = triple_train_val_dataloaders(
        datasets,
        batch_size=batch_size,
        num_workers=num_workers,
        use_fraction_of_data=use_fraction_of_data,
    )

    models = [
        resnet18(pretrained=True),
        resnet18(pretrained=True),
        resnet18(pretrained=True)
    ]

    for i, model in enumerate(models):

        ## set up paths
        save_model_path = "{}_{}_state_dict.pkl".format(root, model_names[i])
        save_mat_path_fc = "{}_{}_data_fc.mat".format(root, model_names[i])
        save_mat_path_tune = "{}_{}_data_tune.mat".format(root, model_names[i])

        ## resize last fully connected layer to match our problem
        model.fc = nn.Linear(model.fc.in_features, 17)
        model.type(dtype)

        loss_fn = nn.MultiLabelSoftMarginLoss().type(dtype)

        ## don't load model params from file - instead retrain the model
        if not from_pickle[i]:

            ## first train only last layer
            for param in model.parameters():
                param.requires_grad = False
            for param in model.fc.parameters():
                param.requires_grad = True

            optimizer_1 = optim.Adam(model.fc.parameters(), lr=lr_1,
                                     weight_decay=reg_1)

            train_acc_history_1 = []
            val_acc_history_1 = []
            loss_history_1 = []
            print("training {} fully connected layer".format(model_names[i]))
            for epoch in range(num_epochs_1):
                print("Begin {} epoch {}/{}".format(
                    model_names[i],
                    epoch+1,
                    num_epochs_1
                ))
                epoch_losses, epoch_f2 = train_epoch(train_loaders[i], model,
                                                     loss_fn, optimizer_1,
                                                     dtype, print_every=20,
                                                     sigmoid_threshold=sigmoid_threshold)
                ## f2 score for validation dataset
                f2_acc = validate_epoch(model, val_loaders[i], dtype,
                                        sigmoid_threshold=sigmoid_threshold)
                ## store results
                train_acc_history_1 += epoch_f2
                val_acc_history_1.append(f2_acc)
                loss_history_1 += epoch_losses
                print("END {} epoch {}/{}: Val F2 score = {:.02f}".format(
                    model_names[i],
                    epoch+1,
                    num_epochs_1,
                    f2_acc
                ))

            ## now finetue the whole param set to our data
            for param in model.parameters():
                param.requires_grad = True

            optimizer_2 = optim.Adam(model.parameters(), lr=lr_2,
                                     weight_decay=reg_2)
            scheduler_2 = ReduceLROnPlateau(
                optimizer_2,
                patience=adaptive_lr_patience,
                cooldown=1, verbose=1,
                min_lr=1e-5*lr_2,
                factor=adaptive_lr_factor
            )

            train_acc_history_2 = []
            val_acc_history_2 = []
            loss_history_2 = []
            print("fine tuning {} all layers:".format(model_names[i]))
            for epoch in range(num_epochs_2):
                print("Begin epoch {}/{}".format(epoch+1, num_epochs_2))
                epoch_losses, epoch_f2 = train_epoch(train_loaders[i], model,
                                                     loss_fn, optimizer_2,
                                                     dtype, print_every=20,
                                                     sigmoid_threshold=sigmoid_threshold)
                scheduler_2.step(np.mean(epoch_losses), epoch)
                ## f2 score for validation dataset
                f2_acc = validate_epoch(model, val_loaders[i], dtype,
                                        sigmoid_threshold=sigmoid_threshold)
                ## store results
                train_acc_history_2 += epoch_f2
                val_acc_history_2.append(f2_acc)
                loss_history_2 += epoch_losses
                print("END {} epoch {}/{}: Val F2 score = {:.02f}".format(
                    model_names[i],
                    epoch+1,
                    num_epochs_2,
                    f2_acc
                ))
            ## serialize model data and save as .pkl file
            torch.save(model.state_dict(), save_model_path)
            print("{} model saved as {}".format(model_names[i],
                                            os.path.abspath(save_model_path)))
            ## save loss and accuracy as .mat file
            save_accuracy_and_loss_mat(save_mat_path_fc, train_acc_history_1,
                                       val_acc_history_1, loss_history_1,
                                       num_epochs_1)
            save_accuracy_and_loss_mat(save_mat_path_tune, train_acc_history_2,
                                       val_acc_history_2, loss_history_2,
                                       num_epochs_2)

        ## load model params from file
        else:
            state_dict = torch.load(save_model_path,
                                    map_location=lambda storage, loc: storage)
            model.load_state_dict(state_dict)
            print("{} model loaded from {}".format(model_names[i],
                                            os.path.abspath(save_model_path)))

    ## generate predictions on test data set
    if run_test:
        ## first optize sigmoid thresholds
        print("optimizing sigmoid cutoffs for each class")
        sig_scores, y_array = get_triple_resnet_val_scores(
                                models,
                                train_loaders,
                                dtype,
                                weights=test_model_weights
                            )
        sigmoid_threshold = optimize_F2(sig_scores, y_array)
        print("optimal thresholds: ", sigmoid_threshold)

        print("generating results for test dataset")
        test_loaders = []
        for ip in img_paths:
            test_dataset = ResnetTestDataset(csv_path, ip, dtype)
            test_loaders.append(DataLoader(test_dataset, batch_size=batch_size,
                                           num_workers=num_workers))
        ## use three models to generate predictions
        test_preds = test_triple_resnet(models, test_loaders,
                                        train_loaders[0].dataset.mlb, dtype,
                                        weights=test_model_weights,
                                        sigmoid_threshold=sigmoid_threshold,
                                        out_file_name=test_results_csv_path)
        print("test set results saved as {}".format(
                                        os.path.abspath(test_results_csv_path)))
