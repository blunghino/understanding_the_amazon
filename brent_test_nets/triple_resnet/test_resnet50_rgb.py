"""
testing the basic setup of a model script using a model with two conv layers
"""
import sys
import os.path

from torchvision.models import resnet50
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import np

from training_utils import test_model, get_scores
from read_in_data import (triple_train_val_balance_dataloaders,
                          ResnetTrainDataset, ResnetTestDataset)
from optimize_cutoffs import optimize_F2


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
    root = "triple_resnet50" # name of model
    csv_path = '../../data/train_v2.csv'
    img_paths = [
        '../../data/train-jpg',
        '../../data/train_RinfB',
        '../../data/train_grad'
    ]
    model_names = ['rgb', 'RinfB', 'grad']
    img_ext = '.jpg'
    ## dataloader params
    batch_size = 128
    use_fraction_of_data = 1 # 1 to train on full data set
    ## optimization hyperparams
    sigmoid_threshold = 0.25
    lr_1 = 1e-3
    num_epochs_1 = 3
    reg_1 = 0
    lr_2 = 1e-5
    num_epochs_2 = 18
    reg_2 = 2e-4
    adaptive_lr_patience = 0 # scale lr after loss plateaus for "patience" epochs
    adaptive_lr_factor = 0.1 # scale lr by this factor
    ## whether to generate predictions on test
    run_test = True
    test_csv_path = "../../data/sample_submission_v2.csv"
    test_img_paths = [
        "../../data/test-jpg",
        "../../data/test_RinfB",
        "../../data/test_grad",
    ]
    test_model_weights = (8., 1., 1.)
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

    train_loaders, val_loaders = triple_train_val_balance_dataloaders(
        datasets,
        batch_size=batch_size,
        num_workers=num_workers,
        use_fraction_of_data=use_fraction_of_data,
    )

    models = [
        resnet50(pretrained=True),
        resnet50(pretrained=True),
        resnet50(pretrained=True)
    ]

    i = 0
    model = models[0]

    ## set up paths
    save_model_path = "{}/{}_{}_state_dict.pkl".format(root, root, model_names[i])
    save_mat_path_fc = "{}/{}_{}_data_fc.mat".format(root, root, model_names[i])
    save_mat_path_tune = "{}/{}_{}_data_tune.mat".format(root, root, model_names[i])

    ## resize last fully connected layer to match our problem
    model.fc = nn.Linear(model.fc.in_features, 17)
    model.type(dtype)

    loss_fn = nn.MultiLabelSoftMarginLoss().type(dtype)


    state_dict = torch.load(save_model_path,
                            map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict)
    print("{} model loaded from {}".format(model_names[i],
                                    os.path.abspath(save_model_path)))


    ## first optize sigmoid thresholds
    print("optimizing sigmoid cutoffs for each class")
    sig_scores, y_array = get_scores(models, train_loaders, dtype)
    sigmoid_threshold = optimize_F2(sig_scores, y_array)
    print("optimal thresholds: ", sigmoid_threshold)

    print("generating results for test dataset")
    test_dataset = ResnetTestDataset(csv_path, img_paths[i], dtype)
    test_loaders.append(DataLoader(test_dataset, batch_size=batch_size,
                                       num_workers=num_workers))
    ## use three models to generate predictions
    test_preds = test_model(model, test_loader,
                                    train_loaders[0].dataset.mlb, dtype,
                                    sigmoid_threshold=sigmoid_threshold,
                                    out_file_name=test_results_csv_path)
    print("test set results saved as {}".format(
                                    os.path.abspath(test_results_csv_path)))
