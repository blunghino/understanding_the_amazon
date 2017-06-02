from training_utils import validate_epoch
import os
import numpy as np
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch import np
from layers import Flatten
from read_in_data import generate_train_val_dataloader, AmazonDataset, AmazonTestDataset
from torch.autograd import Variable
from loss import f2_score
# import cPickle as pickle



def get_optimal_cutoffs(save_model_path, model_function, precision = 0.001, csv_path = 'data/train_v2.csv', img_path = 'data/train-jpg', img_ext = '.jpg', dtype = 'torch.cuda.FloatTensor', batch_size = 128, num_workers = 4, verbose = False):
    ## Load model
    model, dataset = model_function(csv_path, img_path, img_ext, dtype)
    state_dict = torch.load(save_model_path,
                            map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict)
    print("model loaded from {}".format(os.path.abspath(save_model_path)))

    ## Generate Loader
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers
    )

    thresholds = 0.5 * np.ones(17)
    num_steps = int(np.ceil(np.log(precision/0.5) / np.log(0.5)))

    sig_scores_array, y_array = get_scores(model, data_loader, dtype, sigmoid_threshold=thresholds)
    # pickle.dump(sig_scores_array, open( "sig_scores_array.pkl", "wb" ))
    # pickle.dump(y_array, open( "y_array.pkl", "wb" ))
    # sig_scores_array = pickle.load(open( "sig_scores_array.pkl", "rb" ))
    # y_array = pickle.load(open( "y_array.pkl", "rb" ))

    originalF2 = get_F2(sig_scores_array, y_array, thresholds)
    for label_index in range(17):
        if verbose:
            print('label index', label_index)
            print(thresholds, '\n')
        step_size = 0.25
        for step in range(num_steps):
            f2_before = get_F2(sig_scores_array, y_array, thresholds)
            if verbose:
                print('F2 = ', f2_before, 'with class cutoff', thresholds[label_index])
                print('step', step)

            
            increased_thersholds = deepcopy(thresholds)
            increased_thersholds[label_index] += step_size
            f2_increased = get_F2(sig_scores_array, y_array, increased_thersholds)
            if verbose:
                print('F2 score if increased: ', f2_increased)

            decreased_thresholds = deepcopy(thresholds)
            decreased_thresholds[label_index] -= step_size
            f2_decreased = get_F2(sig_scores_array, y_array, decreased_thresholds)
            if verbose:
                print('F2 score if decreased: ', f2_decreased)

            if f2_increased > f2_decreased:
                if verbose:
                    print('increasing by ', step_size)
                thresholds = deepcopy(increased_thersholds)
            else:
                if verbose:
                    print('decreasing by ', step_size)
                thresholds = deepcopy(decreased_thresholds)

            step_size *= 0.5

    finalF2 = get_F2(sig_scores_array, y_array, thresholds)
    print('Original F2: ', originalF2, '\nFinal F2: ', finalF2)
    return thresholds

def get_scores(model, loader, dtype, sigmoid_threshold=None):
    """
    
    """
    n_samples = len(loader.sampler)
    if sigmoid_threshold is None:
        sigmoid_threshold = 0.5
    x, y = loader.dataset[0]
    y_array = np.zeros((n_samples, y.size()[0]))
    sig_scores_array = np.zeros(y_array.shape)
    bs = loader.batch_size
    ## Put the model in test mode
    model.eval()
    for i, (x, y) in enumerate(loader):
        x_var = Variable(x.type(dtype), volatile=True)

        scores = model(x_var)

        ## these are the predicted classes
        ## https://discuss.pytorch.org/t/calculating-accuracy-for-a-multi-label-classification-problem/2303
        # y_pred = torch.sigmoid(scores).data.numpy() > sigmoid_threshold
        
        if dtype == 'torch.cuda.FloatTensor':
            sig_scores = torch.sigmoid(scores).data.cpu().numpy()

            y_array[i*bs:(i+1)*bs,:] = y.cpu().numpy()
            sig_scores_array[i*bs:(i+1)*bs,:] = sig_scores
        elif dtype == 'torch.FloatTensor':
            sig_scores = torch.sigmoid(scores).data.numpy()

            y_array[i*bs:(i+1)*bs,:] = y.numpy()
            sig_scores_array[i*bs:(i+1)*bs,:] = sig_scores
        else:
            print('dtype ' + dtype + ' not supported')

    # return f2_score(torch.from_numpy(y_array), torch.from_numpy(y_pred_array))
    return sig_scores_array, y_array


def get_F2(sig_scores_array, y_array, sigmoid_threshold):
    y_pred = 1.*(sig_scores_array > sigmoid_threshold)
    y_array = torch.from_numpy(y_array)
    y_pred = torch.from_numpy(y_pred)
    return f2_score(y_array, y_pred)

def get_conv6_model(csv_path, img_path, img_ext, dtype):
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

    dataset = AmazonDataset(csv_path, img_path, img_ext, dtype)

    return model, dataset


def get_resnet_model(csv_path, img_path, img_ext, dtype):
    from torchvision.models import resnet18
    import torchvision.transforms as T
    model = resnet18()
    model.fc = nn.Linear(model.fc.in_features, 17)
    model.type(dtype)

    transform_list = [
        T.Scale(224)
    ]

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    dataset = AmazonDataset(csv_path, img_path, img_ext, dtype,
                            transform_list=transform_list, three_band=True,
                            channel_means=IMAGENET_MEAN, channel_stds=IMAGENET_STD)

    return model, dataset



if __name__ == '__main__':
    save_model_path = 'resnet18_pretrained_state_dict.pkl'
    model_function = get_resnet_model
    print(get_optimal_cutoffs(save_model_path, model_function, precision = 0.001, csv_path = 'data/train_v2.csv', img_path = 'data/train-jpg', img_ext = '.jpg', dtype = 'torch.cuda.FloatTensor', batch_size = 64, num_workers = 0, verbose = False))


































