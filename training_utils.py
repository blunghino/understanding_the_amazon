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


def train(loader_train, model, loss_fn, optimizer, dtype,
          num_epochs=1, print_every=1e5):
    """
    train `model` on data from `loader_train`

    inputs:
    `loader_train` object subclassed from torch.data.DataLoader
    `model` neural net, subclassed from torch.nn.Module
    `loss_fn` loss function see torch.nn for examples
    `optimizer` subclassed from torch.optim.Optimizer
    `dtype` data type for variables
        eg torch.FloatTensor (cpu) or torch.cuda.FloatTensor (gpu)

    from cs231n assignment 2
    """
    for epoch in range(num_epochs):
        print('Starting epoch %d / %d' % (epoch + 1, num_epochs))
        model.train()
        for t, (x, y) in enumerate(loader_train):
            x_var = Variable(x.type(dtype))
            y_var = Variable(y.type(dtype))

            scores = model(x_var)

            loss = loss_fn(scores, y_var)
            if (t + 1) % print_every == 0:
                print('t = %d, loss = %.4f' % (t + 1, loss.data[0]))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

def train_epoch(loader_train, model, loss_fn, optimizer, dtype, print_every=20):
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
    ## acc_history = []
    loss_history = []
    model.train()
    for t, (x, y) in enumerate(loader_train):
        x_var = Variable(x.type(dtype))
        y_var = Variable(y.type(dtype))

        scores = model(x_var)

        loss = loss_fn(scores, y_var)
        loss_history.append(loss.data[0])

        ## todo output training accuracy for each batch???

        if (t + 1) % print_every == 0:
            print('t = %d, loss = %.4f' % (t + 1, loss.data[0]))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss_history

def validate_epoch(model, loader, dtype):
    """
    validation for MultiLabelMarginLoss using f2 score

    `model` is a trained subclass of torch.nn.Module
    `loader` is a torch.dataset.DataLoader for validation data
    `dtype` data type for variables
        eg torch.FloatTensor (cpu) or torch.cuda.FloatTensor (gpu)
    """
    x, y = loader.dataset[0]
    y_array = torch.zeros((len(loader.sampler), y.size()[0]))
    y_pred_array = torch.zeros(y_array.size())
    bs = loader.batch_size
    ## Put the model in test mode
    model.eval()
    for i, (x, y) in enumerate(loader):
        x_var = Variable(x.type(dtype), volatile=True)

        scores = model(x_var)

        ## these are the predicted classes
        ## https://discuss.pytorch.org/t/calculating-accuracy-for-a-multi-label-classification-problem/2303
        y_pred = torch.sigmoid(scores).data > 0.5

        y_array[i*bs:(i+1)*bs,:] = y
        y_pred_array[i*bs:(i+1)*bs,:] = y_pred

    return f2_score(y_array, y_pred_array)

def test_model(model, loader, dtype, out_file_name="", n_classes=17):
    """
    run the model on test data and generate a csv file for submission to kaggle
    """
    y_pred_array = torch.zeros((len(loader.sampler), n_classes))
    file_names=[]
    bs = loader.batch_size
    ## Put the model in test mode
    model.eval()
    for i, (x, file_name) in enumerate(loader):
        x_var = Variable(x.type(dtype), volatile=True)
        file_names.append(file_name)
        scores = model(x_var)

        ## https://discuss.pytorch.org/t/calculating-accuracy-for-a-multi-label-classification-problem/2303
        y_pred = torch.sigmoid(scores).data > 0.5
        y_pred_array[i*bs:(i+1)*bs,:] = y_pred

    ## generate labels from MultiLabelBinarizer
    labels = loader.dataset.mlb.inverse_transform(y_pred_array)
    ## write output file
    if out_file_name:
        with open(out_file_name, 'w', newline='') as csvfile:
            fieldnames = ['image_name', 'tags']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for i in range(len(labels)):
                str1=""
                for j in range(n_classes):
                    if(y_pred_array[i,j]==1):
                        str1+=str(labels[j])+" "
                writer.writerow({'image_name': file_names[i], 'tags': str1})
    return y_pred_array
