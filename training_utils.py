"""
boiler plate code to run simple training examples.
Also analysis code to be reused (eg plotting functions)
some code from cs231n assignment 2 pytorch ipynb
"""
import torch
from torch import np
from torch.autograd import Variable
import csv
from loss import f2_score


def train_epoch(loader_train, model, loss_fn, optimizer, dtype,
                sigmoid_threshold=None, print_every=20):
    """
    train `model` on data from `loader_train` for one epoch

    inputs:
    `loader_train` object subclassed from torch.data.DataLoader
    `model` neural net, subclassed from torch.nn.Module
    `loss_fn` loss function see torch.nn for examples
    `optimizer` subclassed from torch.optim.Optimizer
    `dtype` data type for variables
        eg torch.FloatTensor (cpu) or torch.cuda.FloatTensor (gpu)
    """
    acc_history = []
    loss_history = []
    model.train()
    for t, (x, y) in enumerate(loader_train):
        x_var = Variable(x.type(dtype))
        y_var = Variable(y.type(dtype))

        scores = model(x_var)

        loss = loss_fn(scores, y_var)
        loss_history.append(loss.data[0])

        y_pred = torch.sigmoid(scores).data > sigmoid_threshold
        acc = f2_score(y, y_pred)
        acc_history.append(acc)

        if (t + 1) % print_every == 0:
            print('t = %d, loss = %.4f, f2 = %.4f' % (t + 1, loss.data[0], acc))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss_history, acc_history

def validate_epoch(model, loader, dtype, sigmoid_threshold=None):
    """
    validation for MultiLabelMarginLoss using f2 score

    `model` is a trained subclass of torch.nn.Module
    `loader` is a torch.dataset.DataLoader for validation data
    `dtype` data type for variables
        eg torch.FloatTensor (cpu) or torch.cuda.FloatTensor (gpu)
    """
    n_samples = len(loader.sampler)
    if sigmoid_threshold is None:
        sigmoid_threshold = 0.5
    x, y = loader.dataset[0]
    y_array = np.zeros((n_samples, y.size()[0]))
    y_pred_array = np.zeros(y_array.shape)
    bs = loader.batch_size
    ## Put the model in test mode
    model.eval()
    for i, (x, y) in enumerate(loader):
        x_var = Variable(x.type(dtype), volatile=True)

        scores = model(x_var)

        ## these are the predicted classes
        ## https://discuss.pytorch.org/t/calculating-accuracy-for-a-multi-label-classification-problem/2303
        y_pred = torch.sigmoid(scores).data.numpy() > sigmoid_threshold

        y_array[i*bs:(i+1)*bs,:] = y.numpy()
        y_pred_array[i*bs:(i+1)*bs,:] = y_pred

    return f2_score(torch.from_numpy(y_array), torch.from_numpy(y_pred_array))

def test_model(model, loader, mlb, dtype, out_file_name="",
               sigmoid_threshold=None, n_classes=17):
    """
    run the model on test data and generate a csv file for submission to kaggle
    """
    if sigmoid_threshold is None:
        sigmoid_threshold = 0.5
    y_pred_array = np.zeros((len(loader.sampler), n_classes))
    file_names = []
    bs = loader.batch_size
    ## Put the model in test mode
    model.eval()
    for i, (x, file_name) in enumerate(loader):
        x_var = Variable(x.type(dtype), volatile=True)
        file_names += list(file_name)
        scores = model(x_var)

        ## https://discuss.pytorch.org/t/calculating-accuracy-for-a-multi-label-classification-problem/2303
        y_pred = torch.sigmoid(scores).data.numpy() > sigmoid_threshold

        y_pred_array[i*bs:(i+1)*bs,:] = y_pred

    ## generate labels from MultiLabelBinarizer
    labels = mlb.inverse_transform(y_pred_array)

    ## write output file
    if out_file_name:
        with open(out_file_name, 'w', newline='') as csvfile:
            fieldnames = ['image_name', 'tags']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for i, labs in enumerate(labels):
                str1 = ""
                for lab in labs:
                    ## check if there is a label match
                    str1 += str(lab) + " "

                writer.writerow({'image_name': file_names[i], 'tags': str1})

    return y_pred_array


def test_triple_resnet(models, loaders, mlb, dtype, weights=(1,1,1),
                       out_file_name="", sigmoid_threshold=None, n_classes=17):
    """
    run 3 models on test data and generate a csv file for submission to kaggle
    """
    if sigmoid_threshold is None:
        sigmoid_threshold = 0.5
    ## store scores for all three models
    s = np.zeros((3, len(loaders[0].sampler), n_classes))
    file_names = []
    bs = loaders[0].batch_size
    ## Put the model in test mode
    models[0].eval()
    models[1].eval()
    models[2].eval()
    ## loop over the three models
    for i, (model, loader) in enumerate(zip(models, loaders)):
        for j, (x, file_name) in enumerate(loader):
            x_var = Variable(x.type(dtype), volatile=True)
            if i == 0:
                file_names += list(file_name)
            score_var = model(x_var)
            ## store each set of scores
            if dtype is torch.FloatTensor:
                s[i,j*bs:(j+1)*bs,:] = score_var.data.numpy()
            else:
                s[i,j*bs:(j+1)*bs,:] = score_var.data.cpu().numpy()

    ## weighted average of scores from 3 models
    scores = (weights[0]*s[0,:,:] + weights[1]*s[1,:,:] + weights[2]*s[2,:,:]) / sum(weights)

    ## https://discuss.pytorch.org/t/calculating-accuracy-for-a-multi-label-classification-problem/2303
    y_pred = torch.sigmoid(torch.from_numpy(scores)).numpy() > sigmoid_threshold

    ## generate labels from MultiLabelBinarizer
    labels = mlb.inverse_transform(y_pred)

    ## write output file
    if out_file_name:
        with open(out_file_name, 'w', newline='') as csvfile:
            fieldnames = ['image_name', 'tags']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for i, labs in enumerate(labels):
                str1 = ""
                for lab in labs:
                    ## check if there is a label match
                    str1 += str(lab) + " "

                writer.writerow({'image_name': file_names[i], 'tags': str1})

    return y_pred

def get_triple_resnet_val_scores(models, loaders, dtype,
                                 weights=(1,1,1), n_classes=17):
    """
    given three models and training dataloaders,
    return sigmoid scores and correct labels
    """
    ## store scores for all three models
    s = np.zeros((3, len(loaders[0].sampler), n_classes))
    y_array = np.zeros(s.shape[1:])
    bs = loaders[0].batch_size
    ## Put the model in test mode
    models[0].eval()
    models[1].eval()
    models[2].eval()
    ## loop over the three models
    for i, (model, loader) in enumerate(zip(models, loaders)):
        for j, (x, y) in enumerate(loader):
            x_var = Variable(x.type(dtype), volatile=True)
            if i == 0:
                if dtype is torch.cuda.FloatTensor:
                    y_array[j*bs:(j+1)*bs,:] = y.cpu().numpy()
                else:
                    y_array[j*bs:(j+1)*bs,:] = y.numpy()

            score_var = model(x_var)

            ## store each set of scores
            if dtype is torch.cuda.FloatTensor:
                s[i,j*bs:(j+1)*bs,:] = score_var.data.cpu().numpy()
            elif dtype is torch.FloatTensor:
                s[i,j*bs:(j+1)*bs,:] = score_var.data.numpy()

    ## weighted average of scores from 3 models
    scores = (weights[0]*s[0,:,:] + weights[1]*s[1,:,:] + weights[2]*s[2,:,:]) / sum(weights)

    sig_scores = torch.sigmoid(torch.from_numpy(scores)).numpy()

    return sig_scores, y_array