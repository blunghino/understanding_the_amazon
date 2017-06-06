"""
testing the basic setup of a model script using a model with two conv layers
"""
import sys
import os.path

from torchvision.models import resnet101
import torchvision.transforms as T
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch import np

from pytorch_addons.pytorch_lr_scheduler.lr_scheduler import ReduceLROnPlateau

from training_utils import train_epoch, validate_epoch, test_model
from read_in_data import generate_train_val_dataloader, AmazonDataset, AmazonTestDataset
from plotting_tools import save_accuracy_and_loss_mat
from optimize_cutoffs import optimize_F2, get_scores

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

    root = "resnet52_pretrained" # name of model
    save_model_path = "{}_state_dict.pkl".format(root)
    save_mat_path_fc = "{}_loss_and_acc_fc.mat".format(root)
    save_mat_path_tune = "{}_loss_and_acc_tune.mat".format(root)
    csv_path = '../../data/train_v2.csv'
    img_path = '../../data/train-jpg'
    img_ext = '.jpg'
    ## dataloader params
    batch_size = 32
    use_fraction_of_data = 1 # 1 to train on full data set
    ## optimization hyperparams
    lr_1 = 1e-3
    num_epochs_1 = 4
    reg_1 = 0
    lr_2 = 5e-5
    num_epochs_2 = 16
    reg_2 = 5e-4
    adaptive_lr_patience = 0 # scale lr after loss plateaus for "patience" epochs
    adaptive_lr_factor = 0.1 # scale lr by this factor
    ## whether to generate predictions on test
    run_test = True
    test_csv_path = "../../data/sample_submission_v2.csv"
    test_img_path = "../../data/test-jpg"
    test_results_csv_path = "{}_results.csv".format(root)
    ############################################################################

    ## cpu/gpu setup
    if use_cuda:
        dtype = torch.cuda.FloatTensor
        num_workers = 0
    else:
        dtype = torch.FloatTensor
        num_workers = 4

    transform_list = [
        T.Scale(224)
    ]

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    dataset = AmazonDataset(csv_path, img_path, img_ext, dtype,
                            transform_list=transform_list, three_band=True,
                            channel_means=IMAGENET_MEAN, channel_stds=IMAGENET_STD)

    train_loader, val_loader = generate_train_val_dataloader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        use_fraction_of_data=use_fraction_of_data,
    )

    ## pretrained resnet
    model = resnet101(pretrained=True)
    ## resize last fully connected layer to match our problem
    model.fc = nn.Linear(model.fc.in_features, 17)
    model.type(dtype)

    loss_fn = nn.MultiLabelSoftMarginLoss().type(dtype)

    ## don't load model params from file - instead retrain the model
    if not from_pickle:

        ## first train only last layer
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True

        optimizer_1 = optim.Adam(model.fc.parameters(), lr=lr_1, weight_decay=reg_1)

        train_acc_history_1 = []
        val_acc_history_1 = []
        loss_history_1 = []
        print("training final fully connected layer")
        for epoch in range(num_epochs_1):
            print("Begin epoch {}/{}".format(epoch+1, num_epochs_1))
            epoch_losses, epoch_f2 = train_epoch(train_loader, model, loss_fn,
                                                 optimizer_1, dtype, print_every=10)
            ## f2 score for validation dataset
            f2_acc = validate_epoch(model, val_loader, dtype)
            ## store results
            train_acc_history_1 += epoch_f2
            val_acc_history_1.append(f2_acc)
            loss_history_1 += epoch_losses
            print("END epoch {}/{}: validation F2 score = {:.02f}".format(epoch+1, num_epochs_1, f2_acc))

        ## now finetue the whole param set to our data
        for param in model.parameters():
            param.requires_grad = True

        optimizer_2 = optim.Adam(model.parameters(), lr=lr_2, weight_decay=reg_2)
        scheduler_2 = ReduceLROnPlateau(optimizer_2, patience=adaptive_lr_patience,
                                          cooldown=1, verbose=1, min_lr=1e-5*lr_2,
                                          factor=adaptive_lr_factor)

        train_acc_history_2 = []
        val_acc_history_2 = []
        loss_history_2 = []
        print("fine tuning all layers:")
        for epoch in range(num_epochs_2):
            print("Begin epoch {}/{}".format(epoch+1, num_epochs_2))
            epoch_losses, epoch_f2 = train_epoch(train_loader, model, loss_fn,
                                                 optimizer_2, dtype, print_every=10)
            scheduler_2.step(np.mean(epoch_losses), epoch)
            ## f2 score for validation dataset
            f2_acc = validate_epoch(model, val_loader, dtype)
            ## store results
            train_acc_history_2 += epoch_f2
            val_acc_history_2.append(f2_acc)
            loss_history_2 += epoch_losses
            print("END epoch {}/{}: validation F2 score = {:.02f}".format(epoch+1, num_epochs_2, f2_acc))
        ## serialize model data and save as .pkl file
        torch.save(model.state_dict(), save_model_path)
        print("model saved as {}".format(os.path.abspath(save_model_path)))
        ## save loss and accuracy as .mat file
        save_accuracy_and_loss_mat(save_mat_path_fc, train_acc_history_1,
                                   val_acc_history_1, loss_history_1, num_epochs_1)
        save_accuracy_and_loss_mat(save_mat_path_tune, train_acc_history_2,
                                   val_acc_history_2, loss_history_2, num_epochs_2)

    ## load model params from file
    else:
        state_dict = torch.load(save_model_path,
                                map_location=lambda storage, loc: storage)
        model.load_state_dict(state_dict)
        print("model loaded from {}".format(os.path.abspath(save_model_path)))

    ## generate predictions on test data set
    if run_test:
	sig_scores,y_array=get_scores(model,train_loader,dtype)
	sigmoid_threshold=optimize_F2(sig_scores,y_array)
        test_dataset = AmazonTestDataset(csv_path, img_path, img_ext, dtype,
                        three_band=True, transform_list=transform_list,
                        channel_means=IMAGENET_MEAN, channel_stds=IMAGENET_STD)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)
        test_preds = test_model(model, test_loader, train_loader.dataset.mlb, dtype,sigmoid_threshold=sigmoid_threshold,
                                out_file_name=test_results_csv_path)
        print("test set results saved as {}".format(os.path.abspath(test_results_csv_path)))
